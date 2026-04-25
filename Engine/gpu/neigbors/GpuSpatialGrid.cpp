#include "GpuSpatialGrid.h"

#include <array>
#include <cassert>

#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/gpu/neigbors/GpuGridBuffers.h"
#include "generated/shaders/spatial_grid.wgsl.h"

void GpuSpatialGrid::buildPipelines() {
    auto& ctx = WGPUContext::instance();

    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(spatial_gridWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.label = wgpu::StringView("SpatialGridShader");
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = ctx.device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 7> bglEntries{};

    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Compute;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.hasDynamicOffset = true;
    bglEntries[0].buffer.minBindingSize = sizeof(Uniforms);

    auto makeStorageEntry = [](uint32_t binding, wgpu::BufferBindingType type) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        return e;
    };
    bglEntries[1] = makeStorageEntry(1, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[2] = makeStorageEntry(2, wgpu::BufferBindingType::Storage);
    bglEntries[3] = makeStorageEntry(3, wgpu::BufferBindingType::Storage);
    bglEntries[4] = makeStorageEntry(4, wgpu::BufferBindingType::Storage);
    bglEntries[5] = makeStorageEntry(5, wgpu::BufferBindingType::Storage);
    bglEntries[6] = makeStorageEntry(6, wgpu::BufferBindingType::Storage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("SpatialGrid_BGL");
    bglDesc.entryCount = bglEntries.size();
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = ctx.device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("SpatialGrid_PipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = ctx.device().createPipelineLayout(plDesc);

    auto makePipeline = [&](std::string_view entry, std::string_view label) {
        wgpu::ComputePipelineDescriptor d{};
        d.label = wgpu::StringView(label);
        d.layout = pipelineLayout_;
        d.compute.module = shaderModule_;
        d.compute.entryPoint = wgpu::StringView(entry);
        return ctx.device().createComputePipeline(d);
    };

    pipeline_count_ = makePipeline("count", "GridPipeline_Count");
    pipeline_scanBlocks_ = makePipeline("scan_blocks", "GridPipeline_ScanBlocks");
    pipeline_scanLevel2_ = makePipeline("scan_level2", "GridPipeline_ScanLevel2");
    pipeline_addOffsets_ = makePipeline("add_offsets", "GridPipeline_AddOffsets");
    pipeline_sort_ = makePipeline("sort", "GridPipeline_Sort");
}

void GpuSpatialGrid::prepare(const GpuGridBuffers& gridBufs, wgpu::Buffer bufPos) {
    const size_t nAtoms = gridBufs.countAtoms();
    const size_t nCells = gridBufs.countCells();
    const size_t blockCount = (nCells + kWorkgroupScan - 1) / kWorkgroupScan;

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = size;
        return e;
    };

    std::array<wgpu::BindGroupEntry, 7> entries{};
    entries[0] = makeBE(0, sharedUniforms_, sizeof(Uniforms));
    entries[1] = makeBE(1, bufPos, nAtoms * 4 * sizeof(float));
    entries[2] = makeBE(2, gridBufs.bufCount(), nCells * sizeof(uint32_t));
    entries[3] = makeBE(3, gridBufs.bufOffsets(), (nCells + 1) * sizeof(uint32_t));
    entries[4] = makeBE(4, gridBufs.bufReorderCursors(), nCells * sizeof(uint32_t));
    entries[5] = makeBE(5, gridBufs.bufAtomsInCells(), nAtoms * sizeof(uint32_t));
    entries[6] = makeBE(6, gridBufs.bufBlockSums(), blockCount * sizeof(uint32_t));

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("SpatialGrid_BindGroup");
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    bindGroup_ = WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuSpatialGrid::runPass(wgpu::CommandEncoder enc, wgpu::ComputePipeline pipeline, uint32_t workgroupsX, std::string_view label) const {
    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView(label);
    auto pass = enc.beginComputePass(pd);
    pass.setPipeline(pipeline);

    uint32_t dynOffset = kUniformOffset;
    pass.setBindGroup(0, bindGroup_, 1, &dynOffset);
    pass.dispatchWorkgroups(workgroupsX, 1, 1);
    pass.end();
}

void GpuSpatialGrid::record(wgpu::CommandEncoder enc, const World& world) {
    assert(isReady());
    assert(bindGroup_ != nullptr);

    const Vec3u& gs = world.getGridSize();
    const uint32_t nAtoms = static_cast<uint32_t>(world.getAtomBuffers().countAtoms());
    const uint32_t nCells = gs.x * gs.y * gs.z;

    const uint32_t workgroupsAtoms = (nAtoms + kWorkgroupCount - 1) / kWorkgroupCount;
    const uint32_t blockCount = (nCells + kWorkgroupScan - 1) / kWorkgroupScan;

    enc.clearBuffer(world.getGridBuffers().bufCount(), 0, nCells * sizeof(uint32_t));

    runPass(enc, pipeline_count_, workgroupsAtoms, "grid_count pass");
    runPass(enc, pipeline_scanBlocks_, blockCount, "grid_scan_blocks pass");
    runPass(enc, pipeline_scanLevel2_, 1, "grid_scan_level2 pass");
    runPass(enc, pipeline_addOffsets_, blockCount, "grid_add_offsets pass");
    runPass(enc, pipeline_sort_, workgroupsAtoms, "grid_sort pass");
}
