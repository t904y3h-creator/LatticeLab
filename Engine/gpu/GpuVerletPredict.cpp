#include "GpuVerletPredict.h"

#include "GpuAtomBuffers.h"

#include <array>
#include <cassert>
#include <cstring>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/verlet_predict.wgsl.h"

GpuVerletPredict::GpuVerletPredict() { buildPipeline(); }

void GpuVerletPredict::buildPipeline() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(verlet_predictWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.label = wgpu::StringView("VerletPredictShader");
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 5> bglEntries{};

    auto makeStorageEntry = [](uint32_t binding, wgpu::BufferBindingType type) -> wgpu::BindGroupLayoutEntry {
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
    bglEntries[0].buffer.minBindingSize = 16; // TODO сделать структуру Uniform как часть класса
    bglEntries[1] = makeStorageEntry(1, wgpu::BufferBindingType::Storage);
    bglEntries[2] = makeStorageEntry(2, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[3] = makeStorageEntry(3, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[4] = makeStorageEntry(4, wgpu::BufferBindingType::ReadOnlyStorage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = wgpu::StringView("VerletPredict_BindGroupLayout");
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = bindGroupLayout_;

    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.label = wgpu::StringView("VerletPredict_PipelineLayout");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = WGPUContext::instance().device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.label = wgpu::StringView("VerletPredict_Pipeline");
    cpDesc.layout = pipelineLayout_;
    cpDesc.compute.module = shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = WGPUContext::instance().device().createComputePipeline(cpDesc);

    wgpu::BufferDescriptor ubDesc{};
    ubDesc.label = wgpu::StringView("VerletPredict_Uniforms");
    ubDesc.size = 16;
    ubDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    ubDesc.mappedAtCreation = false;
    uniformBuffer_ = WGPUContext::instance().device().createBuffer(ubDesc);
}

wgpu::BindGroup GpuVerletPredict::makeBindGroup(const GpuAtomBuffers& buffers) const {
    const size_t cap = buffers.countAtoms();
    const size_t vec4Bytes = cap * 4 * sizeof(float);
    const size_t f32Bytes = cap * sizeof(float);

    std::array<wgpu::BindGroupEntry, 5> entries{};

    auto makeStorageBE = [](uint32_t binding, wgpu::Buffer buf, size_t bytes) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = bytes;
        return e;
    };

    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer_;
    entries[0].offset = 0;
    entries[0].size = 16;
    entries[1] = makeStorageBE(1, buffers.bufPos(), vec4Bytes);
    entries[2] = makeStorageBE(2, buffers.bufVel(), vec4Bytes);
    entries[3] = makeStorageBE(3, buffers.bufF(), vec4Bytes);
    entries[4] = makeStorageBE(4, buffers.bufInvMass(), f32Bytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.label = wgpu::StringView("VerletPredict_BindGroup");
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();

    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuVerletPredict::record(wgpu::CommandEncoder& enc, const GpuAtomBuffers& buffers, uint32_t atomCount, float dt) {
    assert(isReady());
    assert(atomCount <= buffers.countAtoms());

    struct GpuUniforms {
        float dt;
        uint32_t atomCount;
        uint32_t pad[2];
    } uni{dt, atomCount, {0u, 0u}};

    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bindGroup = makeBindGroup(buffers);

    wgpu::ComputePassDescriptor passDesc{};
    passDesc.label = wgpu::StringView("VerletPredict pass");
    wgpu::ComputePassEncoder pass = enc.beginComputePass(passDesc);

    pass.setPipeline(pipeline_);
    pass.setBindGroup(0, bindGroup, 0, nullptr);

    const uint32_t groups = (atomCount + kWorkgroupSize - 1) / kWorkgroupSize;
    pass.dispatchWorkgroups(groups, 1, 1);
    pass.end();
}