#include "GpuVerletCorrect.h"

#include "GpuAtomBuffers.h"

#include <array>
#include <cassert>

#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/verlet_correct.wgsl.h"

void GpuVerletCorrect::buildPipeline() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(verlet_correctWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.label = wgpu::StringView("VerletCorrectShader");
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 5> bglEntries{};

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
    bglEntries[1] = makeStorageEntry(1, wgpu::BufferBindingType::Storage);
    bglEntries[2] = makeStorageEntry(2, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[3] = makeStorageEntry(3, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[4] = makeStorageEntry(4, wgpu::BufferBindingType::ReadOnlyStorage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("VerletCorrect_BindGroupLayout");
    bglDesc.entryCount = bglEntries.size();
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = *bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("VerletCorrect_PipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = WGPUContext::instance().device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.label = wgpu::StringView("VerletCorrect_Pipeline");
    cpDesc.layout = *pipelineLayout_;
    cpDesc.compute.module = *shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = WGPUContext::instance().device().createComputePipeline(cpDesc);
}

void GpuVerletCorrect::prepare(const GpuAtomBuffers& buffers) {
    const size_t vec4Bytes = buffers.countAtoms() * 4 * sizeof(float);
    const size_t f32Bytes = buffers.countAtoms() * sizeof(float);

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size, size_t offset = 0) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = offset;
        e.size = size;
        return e;
    };

    std::array<wgpu::BindGroupEntry, 5> entries{};
    entries[0] = makeBE(0, sharedUniforms_, sizeof(Uniforms));
    entries[1] = makeBE(1, buffers.bufVel(), vec4Bytes);
    entries[2] = makeBE(2, buffers.bufF(), vec4Bytes);
    entries[3] = makeBE(3, buffers.bufPrevF(), vec4Bytes);
    entries[4] = makeBE(4, buffers.bufInvMass(), f32Bytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("VerletCorrect_BindGroup");
    bgDesc.layout = *bindGroupLayout_;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    bindGroup_ = WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuVerletCorrect::record(wgpu::CommandEncoder& enc, uint32_t atomCount) {
    assert(isReady());
    assert(*bindGroup_ != nullptr);

    wgpu::ComputePassDescriptor passDesc{};
    passDesc.label = wgpu::StringView("VerletCorrect pass");
    wgpu::raii::ComputePassEncoder pass(enc.beginComputePass(passDesc));
    pass->setPipeline(*pipeline_);

    pass->setBindGroup(0, *bindGroup_, 1, &kUniformOffset);
    pass->dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass->end();
}
