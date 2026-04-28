#include "GpuWallForceField.h"

#include <array>
#include <cassert>

#include <webgpu/webgpu-raii.hpp>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/force_wall.wgsl.h"

void GpuWallForceField::buildPipeline() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(force_wallWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 3> bglEntries{};

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
    bglEntries[1] = makeStorageBGLE(1, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[2] = makeStorageBGLE(2, wgpu::BufferBindingType::Storage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("Wall_BGL");
    bglDesc.entryCount = bglEntries.size();
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = *bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("Wall_PipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = WGPUContext::instance().device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.label = wgpu::StringView("Wall_Pipeline");
    cpDesc.layout = *pipelineLayout_;
    cpDesc.compute.module = *shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = WGPUContext::instance().device().createComputePipeline(cpDesc);
}

void GpuWallForceField::prepare(const GpuAtomBuffers& atomBufs) {
    const size_t vec4Bytes = atomBufs.countAtoms() * 4 * sizeof(float);

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = size;
        return e;
    };

    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0] = makeBE(0, sharedUniforms_, sizeof(Uniforms));
    entries[1] = makeBE(1, atomBufs.bufPos(), vec4Bytes);
    entries[2] = makeBE(2, atomBufs.bufF(), vec4Bytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("Wall_BindGroup");
    bgDesc.layout = *bindGroupLayout_;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    bindGroup_ = WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuWallForceField::record(wgpu::CommandEncoder enc, uint32_t atomCount) {
    assert(isReady());
    assert(*bindGroup_ != nullptr);

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("WallForces pass");
    wgpu::raii::ComputePassEncoder pass(enc.beginComputePass(pd));
    pass->setPipeline(*pipeline_);

    uint32_t dynOffset = kUniformOffset;
    pass->setBindGroup(0, *bindGroup_, 1, &dynOffset);
    pass->dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass->end();
}
