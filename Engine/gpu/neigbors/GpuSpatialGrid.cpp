#include "GpuSpatialGrid.h"

#include <array>
#include <cassert>

#include <webgpu/webgpu.hpp>

#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/gpu/neigbors/GpuGridBuffers.h"
#include "generated/shaders/spatial_grid.wgsl.h"

struct GridUniforms {
    float cellSize;
    uint32_t dx;
    uint32_t dy;
    uint32_t dz;
    uint32_t n;
    uint32_t _pad;
};
static_assert(sizeof(GridUniforms) == 24);

GpuSpatialGrid::GpuSpatialGrid() { buildPipelines(); }

void GpuSpatialGrid::buildPipelines() {
    auto& ctx = WGPUContext::instance();

    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(spatial_gridWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.label = wgpu::StringView("SpatialGridShader");
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = ctx.device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 7> bglEntries{};

    auto makeStorageEntry = [](uint32_t binding, wgpu::BufferBindingType type) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        e.buffer.minBindingSize = 0;
        return e;
    };

    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Compute;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = sizeof(GridUniforms);

    bglEntries[1] = makeStorageEntry(1, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[2] = makeStorageEntry(2, wgpu::BufferBindingType::Storage);
    bglEntries[3] = makeStorageEntry(3, wgpu::BufferBindingType::Storage);
    bglEntries[4] = makeStorageEntry(4, wgpu::BufferBindingType::Storage);
    bglEntries[5] = makeStorageEntry(5, wgpu::BufferBindingType::Storage);
    bglEntries[6] = makeStorageEntry(6, wgpu::BufferBindingType::Storage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("SpatialGridBindGroupLayout");
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = ctx.device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("SpatialGridPipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = ctx.device().createPipelineLayout(plDesc);

    auto makePipeline = [&](std::string_view entryPoint, std::string_view label) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor cpDesc{};
        cpDesc.label = wgpu::StringView(label);
        cpDesc.layout = pipelineLayout_;
        cpDesc.compute.module = shaderModule_;
        cpDesc.compute.entryPoint = wgpu::StringView(entryPoint);
        return ctx.device().createComputePipeline(cpDesc);
    };

    pipeline_count_ = makePipeline("count", "GridPipelineCount");
    pipeline_scanBlocks_ = makePipeline("scan_blocks", "GridPipelineScanBlocks");
    pipeline_scanLevel2_ = makePipeline("scan_level2", "GridPipelineScanLevel2");
    pipeline_addOffsets_ = makePipeline("add_offsets", "GridPipelineAddOffsets");
    pipeline_sort_ = makePipeline("sort", "GridPipelineSort");

    uniformBuffer_ = ctx.createBuffer(sizeof(GridUniforms), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "GridUniforms");
}

wgpu::BindGroup GpuSpatialGrid::makeBindGroup(const GpuGridBuffers& gridBufs, wgpu::Buffer bufPos) const {
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
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer_;
    entries[0].offset = 0;
    entries[0].size = sizeof(GridUniforms);

    entries[1] = makeBE(1, bufPos, nAtoms * 4 * sizeof(float));
    entries[2] = makeBE(2, gridBufs.bufCount(), nCells * sizeof(uint32_t));
    entries[3] = makeBE(3, gridBufs.bufOffsets(), (nCells + 1) * sizeof(uint32_t));
    entries[4] = makeBE(4, gridBufs.bufReorderCursors(), nCells * sizeof(uint32_t));
    entries[5] = makeBE(5, gridBufs.bufAtomsInCells(), nAtoms * sizeof(uint32_t));
    entries[6] = makeBE(6, gridBufs.bufBlockSums(), blockCount * sizeof(uint32_t));

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("GridBindGroup");
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();

    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuSpatialGrid::runPass(wgpu::CommandEncoder enc, wgpu::ComputePipeline pipeline, wgpu::BindGroup bindGroup, uint32_t workgroupsX,
                             std::string_view label) const {
    wgpu::ComputePassDescriptor passDesc{};
    passDesc.label = wgpu::StringView(label);
    wgpu::ComputePassEncoder pass = enc.beginComputePass(passDesc);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup, 0, nullptr);
    pass.dispatchWorkgroups(workgroupsX, 1, 1);
    pass.end();
}

void GpuSpatialGrid::record(wgpu::CommandEncoder enc, const World& world) {
    assert(isReady());

    const Vec3u& gs = world.getGridSize();
    const uint32_t nAtoms = static_cast<uint32_t>(world.getAtomBuffers().countAtoms());

    GridUniforms uni{};
    uni.cellSize = world.getGridCellSize();
    uni.dx = gs.x;
    uni.dy = gs.y;
    uni.dz = gs.z;
    uni.n = nAtoms;

    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    const GpuGridBuffers& gridBufs = world.getGridBuffers();

    wgpu::BindGroup bg = makeBindGroup(gridBufs, world.getAtomBuffers().bufPos());

    enc.clearBuffer(gridBufs.bufCount(), 0, gridBufs.countCells() * sizeof(uint32_t));

    const uint32_t workgroupsAtoms = (nAtoms + kWorkgroupCount - 1) / kWorkgroupCount;
    const uint32_t blockCount = (gs.x * gs.y * gs.z + kWorkgroupScan - 1) / kWorkgroupScan;

    runPass(enc, pipeline_count_, bg, workgroupsAtoms, "grid_count");
    runPass(enc, pipeline_scanBlocks_, bg, blockCount, "grid_scan_blocks");
    runPass(enc, pipeline_scanLevel2_, bg, 1, "grid_scan_level2");
    runPass(enc, pipeline_addOffsets_, bg, blockCount, "grid_add_offsets");
    runPass(enc, pipeline_sort_, bg, workgroupsAtoms, "grid_sort");
}
