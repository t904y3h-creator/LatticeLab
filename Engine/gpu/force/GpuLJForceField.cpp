#include "GpuLJForceField.h"

#include <array>
#include <cassert>
#include <vector>

#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/force_lj.wgsl.h"

struct LJUniforms {
    uint32_t atomCount;
    uint32_t typeCount;
    uint32_t grid_dx;
    uint32_t grid_dy;
    uint32_t grid_dz;
    float grid_cell_size;
    uint32_t _pad[2];
};
static_assert(sizeof(LJUniforms) == 32);

void GpuLJForceField::init(const LJTable& ljTable) {
    buildPipeline();
    uploadLJTable(ljTable);
}

void GpuLJForceField::buildPipeline() {
    auto& ctx = WGPUContext::instance();

    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(force_ljWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.label = wgpu::StringView("LJForceFieldShader");
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = ctx.device().createShaderModule(smDesc);

    auto makeStorageBGLE = [](uint32_t binding, wgpu::BufferBindingType type) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        e.buffer.minBindingSize = 0;
        return e;
    };

    std::array<wgpu::BindGroupLayoutEntry, 8> bglEntries{};
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Compute;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = sizeof(LJUniforms);

    bglEntries[1] = makeStorageBGLE(1, wgpu::BufferBindingType::ReadOnlyStorage); // bufPos
    bglEntries[2] = makeStorageBGLE(2, wgpu::BufferBindingType::Storage);         // bufF
    bglEntries[3] = makeStorageBGLE(3, wgpu::BufferBindingType::Storage);         // bufPe
    bglEntries[4] = makeStorageBGLE(4, wgpu::BufferBindingType::ReadOnlyStorage); // bufAtomType
    bglEntries[5] = makeStorageBGLE(5, wgpu::BufferBindingType::ReadOnlyStorage); // ljTableBuffer
    bglEntries[6] = makeStorageBGLE(6, wgpu::BufferBindingType::ReadOnlyStorage); // bufOffsets
    bglEntries[7] = makeStorageBGLE(7, wgpu::BufferBindingType::ReadOnlyStorage); // bufAtomsInCells

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("LJForceFieldBGL");
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = ctx.device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = ctx.device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pipelineLayout_;
    cpDesc.compute.module = shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = ctx.device().createComputePipeline(cpDesc);

    uniformBuffer_ = ctx.createBuffer(sizeof(LJUniforms), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "LJ_Uniforms");
}

void GpuLJForceField::uploadLJTable(const LJTable& ljTable) {
    constexpr size_t n = static_cast<size_t>(AtomData::Type::COUNT);
    std::vector<Vec2f> table(n * n);

    for (size_t i = 0; i < n; ++i) {
        const auto& row = ljTable.pairRow(static_cast<AtomData::Type>(i));
        for (size_t j = 0; j < n; ++j) {
            size_t ind = i * n + j;
            table[ind].x = row[j].potentialC6;
            table[ind].y = row[j].potentialC12;
        }
    }

    const size_t bytes = table.size() * sizeof(float);
    ljTableBuffer_ = WGPUContext::instance().createBuffer(bytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, "LJ_Table");
    WGPUContext::instance().queue().writeBuffer(ljTableBuffer_, 0, table.data(), bytes);
}

wgpu::BindGroup GpuLJForceField::makeBindGroup(const GpuAtomBuffers& atomBufs, const GpuGridBuffers& gridBufs) const {
    const size_t cap = atomBufs.countAtoms();
    const size_t vec4Bytes = cap * 4 * sizeof(float);
    const size_t u32Bytes = cap * sizeof(uint32_t);
    const size_t tableBytes = static_cast<size_t>(AtomData::Type::COUNT) * static_cast<size_t>(AtomData::Type::COUNT) * 2 * sizeof(float);
    const size_t nCells = gridBufs.countCells();

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = size;
        return e;
    };

    std::array<wgpu::BindGroupEntry, 8> entries{};
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer_;
    entries[0].size = sizeof(LJUniforms);

    entries[1] = makeBE(1, atomBufs.bufPos(), vec4Bytes);
    entries[2] = makeBE(2, atomBufs.bufF(), vec4Bytes);
    entries[3] = makeBE(3, atomBufs.bufPe(), cap * sizeof(float));
    entries[4] = makeBE(4, atomBufs.bufAtomType(), u32Bytes);
    entries[5] = makeBE(5, ljTableBuffer_, tableBytes);
    entries[6] = makeBE(6, gridBufs.bufOffsets(), (nCells + 1) * sizeof(uint32_t));
    entries[7] = makeBE(7, gridBufs.bufAtomsInCells(), cap * sizeof(uint32_t));

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();
    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuLJForceField::record(wgpu::CommandEncoder enc, const World& world) {
    assert(isReady());

    const uint32_t atomCount = static_cast<uint32_t>(world.mobileCount());
    const Vec3u& gs = world.getGridSize();

    LJUniforms uni{};
    uni.atomCount = atomCount;
    uni.typeCount = static_cast<uint32_t>(AtomData::Type::COUNT);
    uni.grid_dx = gs.x;
    uni.grid_dy = gs.y;
    uni.grid_dz = gs.z;
    uni.grid_cell_size = world.getGridCellSize();

    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bg = makeBindGroup(world.getAtomBuffers(), world.getGridBuffers());

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("LJ_forces");
    auto pass = enc.beginComputePass(pd);
    pass.setPipeline(pipeline_);
    pass.setBindGroup(0, bg, 0, nullptr);
    pass.dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass.end();
}
