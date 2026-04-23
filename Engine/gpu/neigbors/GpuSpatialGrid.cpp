#include "GpuSpatialGrid.h"

#include <array>
#include <cassert>
#include <cstring>

#include <webgpu/webgpu.hpp>

#include "Engine/NeighborSearch/SpatialGrid.h"
#include "Engine/gpu//neigbors/GpuGridBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/spatial_grid.wgsl.h"

// ─────────────────────────────────────────────────────────────────────────────
// Uniforms (должны совпадать со struct Params в шейдере)
// ─────────────────────────────────────────────────────────────────────────────

struct GridUniforms {
    float cellSize;
    uint32_t dx;
    uint32_t dy;
    uint32_t dz;
    uint32_t n; // atomCount
    uint32_t _pad;
};
static_assert(sizeof(GridUniforms) == 24);

// ─────────────────────────────────────────────────────────────────────────────
// Построение pipeline
// ─────────────────────────────────────────────────────────────────────────────

void GpuSpatialGrid::buildPipelines() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(spatial_gridWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    // 2. Bind group layout
    //   binding 0 — uniform    Params
    //   binding 1 — storage r  pos          (vec4f атомов)
    //   binding 2 — storage rw counts       (atomic<u32>)
    //   binding 3 — storage rw starts       (u32)
    //   binding 4 — storage rw off          (atomic<u32>)
    //   binding 5 — storage rw sortedIdx    (u32)
    //   binding 6 — storage rw blockSums    (u32)

    std::array<wgpu::BindGroupLayoutEntry, 7> bglEntries{};

    auto makeStorageEntry = [](uint32_t binding, wgpu::BufferBindingType type, wgpu::ShaderStage vis = wgpu::ShaderStage::Compute) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = vis;
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
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    // 3. Pipeline layout
    WGPUBindGroupLayout rawBGL = bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = WGPUContext::instance().device().createPipelineLayout(plDesc);

    // 4. Pipelines — по одному на каждый entry point шейдера
    auto makePipeline = [&](std::string_view entryPoint) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor cpDesc{};
        cpDesc.layout = pipelineLayout_;
        cpDesc.compute.module = shaderModule_;
        cpDesc.compute.entryPoint = wgpu::StringView(entryPoint);
        return WGPUContext::instance().device().createComputePipeline(cpDesc);
    };

    pipeline_count_ = makePipeline("count");
    pipeline_scanBlocks_ = makePipeline("scan_blocks");
    pipeline_scanLevel2_ = makePipeline("scan_level2");
    pipeline_addOffsets_ = makePipeline("add_offsets");
    pipeline_sort_ = makePipeline("sort");

    // 5. Uniform buffer
    wgpu::BufferDescriptor ubDesc{};
    ubDesc.label = wgpu::StringView("GridUniforms");
    ubDesc.size = sizeof(GridUniforms);
    ubDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    ubDesc.mappedAtCreation = false;
    uniformBuffer_ = WGPUContext::instance().device().createBuffer(ubDesc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bind group
// ─────────────────────────────────────────────────────────────────────────────

wgpu::BindGroup GpuSpatialGrid::makeBindGroup(GpuGridBuffers& gridBufs, wgpu::Buffer bufPos) const {
    std::array<wgpu::BindGroupEntry, 7> entries{};

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = size;
        return e;
    };

    // Размеры считаем из countAtoms/countCells хранящихся в gridBufs
    const size_t nAtoms = gridBufs.countAtoms();
    const size_t nCells = gridBufs.countCells();
    const size_t blockCount = (nCells + 255) / 256;

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
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();

    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

// ─────────────────────────────────────────────────────────────────────────────

void GpuSpatialGrid::runPass(wgpu::CommandEncoder& enc, wgpu::ComputePipeline pipeline, wgpu::BindGroup bindGroup, uint32_t workgroupsX,
                             std::string_view label) const {
    wgpu::ComputePassDescriptor passDesc{};
    passDesc.label = wgpu::StringView(label);
    wgpu::ComputePassEncoder pass = enc.beginComputePass(passDesc);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup, 0, nullptr);
    pass.dispatchWorkgroups(workgroupsX, 1, 1);
    pass.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch — основной публичный метод
// ─────────────────────────────────────────────────────────────────────────────

void GpuSpatialGrid::dispatch(GpuGridBuffers& gridBufs, wgpu::Buffer bufPos, uint32_t atomCount, const SpatialGrid& grid) {
    assert(isReady());

    // Обновляем uniforms
    GridUniforms uni{};
    uni.cellSize = static_cast<float>(grid.cellSize);
    uni.dx = static_cast<uint32_t>(grid.sizeX - 2);
    uni.dy = static_cast<uint32_t>(grid.sizeY - 2);
    uni.dz = static_cast<uint32_t>(grid.sizeZ - 2);
    uni.n = atomCount;
    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    const uint32_t totalCells = uni.dx * uni.dy * uni.dz;
    gridBufs.resize(atomCount, totalCells);

    const uint32_t blockCount = (totalCells + kWorkgroupScan - 1) / kWorkgroupScan;

    wgpu::BindGroup bg = makeBindGroup(gridBufs, bufPos);

    wgpu::CommandEncoder enc = WGPUContext::instance().device().createCommandEncoder({});

    enc.clearBuffer(gridBufs.bufCount(), 0, gridBufs.countCells() * sizeof(uint32_t));

    // Каждый проход — отдельный ComputePassEncoder.
    // WebGPU гарантирует видимость записей в storage буферах
    // только между отдельными compute pass-ами.
    runPass(enc, pipeline_count_, bg, (atomCount + kWorkgroupCount - 1) / kWorkgroupCount, "grid_count");
    runPass(enc, pipeline_scanBlocks_, bg, blockCount, "grid_scan_blocks");
    runPass(enc, pipeline_scanLevel2_, bg, 1, "grid_scan_level2");
    runPass(enc, pipeline_addOffsets_, bg, blockCount, "grid_add_offsets");
    runPass(enc, pipeline_sort_, bg, (atomCount + kWorkgroupCount - 1) / kWorkgroupCount, "grid_sort");

    wgpu::CommandBuffer cmd = enc.finish({});
    WGPUContext::instance().queue().submit(1, &cmd);
    WGPUContext::instance().device().poll(true, nullptr);
}