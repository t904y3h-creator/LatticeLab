#include "GpuVerletCorrect.h"

#include "GpuAtomBuffers.h"

#include <array>
#include <cassert>

#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/verlet_correct.wgsl.h"

GpuVerletCorrect::GpuVerletCorrect() { buildPipeline(); }

void GpuVerletCorrect::release() {
    auto destroyBuf = [](wgpu::Buffer& b) {
        if (b) {
            b.destroy();
            b = nullptr;
        }
    };
    destroyBuf(uniformBuffer_);
    if (pipeline_) {
        pipeline_.release();
        pipeline_ = nullptr;
    }
    if (pipelineLayout_) {
        pipelineLayout_.release();
        pipelineLayout_ = nullptr;
    }
    if (bindGroupLayout_) {
        bindGroupLayout_.release();
        bindGroupLayout_ = nullptr;
    }
    if (shaderModule_) {
        shaderModule_.release();
        shaderModule_ = nullptr;
    }
}

void GpuVerletCorrect::buildPipeline() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(verlet_correctWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
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
    bglEntries[0].buffer.minBindingSize = 16;
    bglEntries[1] = makeStorageEntry(1, wgpu::BufferBindingType::Storage);
    bglEntries[2] = makeStorageEntry(2, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[3] = makeStorageEntry(3, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[4] = makeStorageEntry(4, wgpu::BufferBindingType::ReadOnlyStorage);

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

    // 4. Compute pipeline
    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pipelineLayout_;
    cpDesc.compute.module = shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = WGPUContext::instance().device().createComputePipeline(cpDesc);

    // 5. Uniform buffer (16 байт: dt, accelDamping, atomCount, pad)
    wgpu::BufferDescriptor ubDesc{};
    ubDesc.label = wgpu::StringView("VerletCorrect_Uniforms");
    ubDesc.size = 16;
    ubDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    ubDesc.mappedAtCreation = false;
    uniformBuffer_ = WGPUContext::instance().device().createBuffer(ubDesc);
}

wgpu::BindGroup GpuVerletCorrect::makeBindGroup(GpuAtomBuffers& buffers) const {
    const size_t vec4Bytes = buffers.capacity() * 4 * sizeof(float);
    const size_t f32Bytes = buffers.capacity() * sizeof(float);

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
    entries[1] = makeStorageBE(1, buffers.bufVel(), vec4Bytes);
    entries[2] = makeStorageBE(2, buffers.bufF(), vec4Bytes);
    entries[3] = makeStorageBE(3, buffers.bufPrevF(), vec4Bytes);
    entries[4] = makeStorageBE(4, buffers.bufInvMass(), f32Bytes);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();

    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuVerletCorrect::dispatch(GpuAtomBuffers& buffers, uint32_t atomCount, float dt, float accelDamping) {
    assert(isReady());
    assert(atomCount <= buffers.capacity());

    struct GpuUniforms {
        float dt;
        float accelDamping;
        uint32_t atomCount;
        uint32_t pad;
    } uni{dt, accelDamping, atomCount, 0u};

    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bindGroup = makeBindGroup(buffers);

    wgpu::CommandEncoder enc = WGPUContext::instance().device().createCommandEncoder({});
    {
        wgpu::ComputePassDescriptor passDesc{};
        passDesc.label = wgpu::StringView("VerletCorrect");
        wgpu::ComputePassEncoder pass = enc.beginComputePass(passDesc);

        pass.setPipeline(pipeline_);
        pass.setBindGroup(0, bindGroup, 0, nullptr);

        const uint32_t groups = (atomCount + kWorkgroupSize - 1) / kWorkgroupSize;
        pass.dispatchWorkgroups(groups, 1, 1);
        pass.end();
    }

    wgpu::CommandBuffer cmd = enc.finish({});
    WGPUContext::instance().queue().submit(1, &cmd);
    WGPUContext::instance().device().poll(false, nullptr);
}
