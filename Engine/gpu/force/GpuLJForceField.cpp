#include "GpuLJForceField.h"

#include <array>
#include <cassert>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/gpu/neigbors/GpuGridBuffers.h"
#include "Engine/math/Vec2.h"
#include "Engine/physics/AtomData.h"
#include "Engine/physics/ForceFields/LJTable.h"
#include "generated/shaders/force_lj.wgsl.h"

void GpuLJForceField::buildPipeline() {
    auto& ctx = WGPUContext::instance();

    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(force_ljWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.label = wgpu::StringView("LJForceFieldShader");
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = ctx.device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 8> bglEntries{};

    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Compute;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.hasDynamicOffset = true;
    bglEntries[0].buffer.minBindingSize = sizeof(Uniforms);

    auto makeStorageBGLE = [](uint32_t binding, wgpu::BufferBindingType type) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        return e;
    };
    bglEntries[1] = makeStorageBGLE(1, wgpu::BufferBindingType::ReadOnlyStorage); // bufPos
    bglEntries[2] = makeStorageBGLE(2, wgpu::BufferBindingType::Storage);         // bufF
    bglEntries[3] = makeStorageBGLE(3, wgpu::BufferBindingType::Storage);         // bufPe
    bglEntries[4] = makeStorageBGLE(4, wgpu::BufferBindingType::ReadOnlyStorage); // bufOffsets
    bglEntries[5] = makeStorageBGLE(5, wgpu::BufferBindingType::ReadOnlyStorage); // bufAtomsInCells
    bglEntries[6] = makeStorageBGLE(6, wgpu::BufferBindingType::ReadOnlyStorage); // ljTableBuffer
    bglEntries[7] = makeStorageBGLE(7, wgpu::BufferBindingType::ReadOnlyStorage); // bufAtomType

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("LJ_BGL");
    bglDesc.entryCount = bglEntries.size();
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = ctx.device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = *bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("LJ_PipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = ctx.device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.label = wgpu::StringView("LJ_Pipeline");
    cpDesc.layout = *pipelineLayout_;
    cpDesc.compute.module = *shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = ctx.device().createComputePipeline(cpDesc);
}

void GpuLJForceField::uploadLJTable(const LJTable& ljTable) {
    constexpr size_t n = static_cast<size_t>(AtomData::Type::COUNT);
    std::vector<Vec2f> table(n * n);

    for (size_t i = 0; i < n; ++i) {
        const auto& row = ljTable.pairRow(static_cast<AtomData::Type>(i));
        for (size_t j = 0; j < n; ++j) {
            table[i * n + j] = {row[j].potentialC6, row[j].potentialC12};
        }
    }

    const size_t bytes = table.size() * sizeof(table[0]);
    ljTableBuffer_ = WGPUContext::instance().createBuffer(bytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, "LJ_Table");
    WGPUContext::instance().queue().writeBuffer(*ljTableBuffer_, 0, table.data(), bytes);
}

void GpuLJForceField::prepare(const GpuAtomBuffers& atomBufs, const GpuGridBuffers& gridBufs) {
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
    entries[0] = makeBE(0, sharedUniforms_, sizeof(Uniforms));
    entries[1] = makeBE(1, atomBufs.bufPos(), vec4Bytes);
    entries[2] = makeBE(2, atomBufs.bufF(), vec4Bytes);
    entries[3] = makeBE(3, atomBufs.bufPe(), cap * sizeof(float));
    entries[4] = makeBE(4, gridBufs.bufOffsets(), (nCells + 1) * sizeof(uint32_t));
    entries[5] = makeBE(5, gridBufs.bufAtomsInCells(), u32Bytes);
    entries[6] = makeBE(6, *ljTableBuffer_, tableBytes);
    entries[7] = makeBE(7, atomBufs.bufAtomType(), u32Bytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("LJ_BindGroup");
    bgDesc.layout = *bindGroupLayout_;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    bindGroup_ = WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuLJForceField::record(wgpu::CommandEncoder enc, uint32_t atomCount) {
    assert(isReady());
    assert(*bindGroup_ != nullptr);

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("LJ_forces");
    wgpu::raii::ComputePassEncoder pass(enc.beginComputePass(pd));
    pass->setPipeline(*pipeline_);

    uint32_t dynOffset = kUniformOffset;
    pass->setBindGroup(0, *bindGroup_, 1, &dynOffset);
    pass->dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass->end();
}
