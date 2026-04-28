#include "GpuStepOps.h"

#include <array>
#include <cassert>
#include <span>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/step_ops.wgsl.h"

void GpuStepOps::buildPipelines() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = wgpu::SType::ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(step_opsWGSL);
    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    auto makeStorageBGLE = [](uint32_t binding, wgpu::BufferBindingType type) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        return e;
    };

    auto makeUniformBGLE = [](uint32_t binding, uint64_t minSize) {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = wgpu::BufferBindingType::Uniform;
        e.buffer.hasDynamicOffset = true;
        e.buffer.minBindingSize = minSize;
        return e;
    };

    auto makeBGL = [&](std::span<wgpu::BindGroupLayoutEntry> entries, std::string_view label) {
        wgpu::BindGroupLayoutDescriptor d{};
        d.label = wgpu::StringView(label);
        d.entryCount = entries.size();
        d.entries = entries.data();
        return WGPUContext::instance().device().createBindGroupLayout(d);
    };

    auto makePipelineLayout = [&](wgpu::BindGroupLayout bgl) {
        WGPUBindGroupLayout raw = bgl;
        wgpu::PipelineLayoutDescriptor d{};
        d.bindGroupLayoutCount = 1;
        d.bindGroupLayouts = &raw;
        return WGPUContext::instance().device().createPipelineLayout(d);
    };

    auto makePipeline = [&](wgpu::PipelineLayout layout, std::string_view entry, std::string_view label) {
        wgpu::ComputePipelineDescriptor d{};
        d.label = wgpu::StringView(label);
        d.layout = layout;
        d.compute.module = *shaderModule_;
        d.compute.entryPoint = wgpu::StringView(entry);
        return WGPUContext::instance().device().createComputePipeline(d);
    };

    // confine
    {
        std::array<wgpu::BindGroupLayoutEntry, 3> entries{
            makeUniformBGLE(0, sizeof(ConfineUniforms)),
            makeStorageBGLE(1, wgpu::BufferBindingType::Storage),
            makeStorageBGLE(2, wgpu::BufferBindingType::Storage),
        };
        bgl_confine_ = makeBGL(entries, "confine_BGL");
        layout_confine_ = makePipelineLayout(*bgl_confine_);
        pipeline_confine_ = makePipeline(*layout_confine_, "confine_to_box", "confine_Pipeline");
    }

    // velcap
    {
        std::array<wgpu::BindGroupLayoutEntry, 2> entries{
            makeUniformBGLE(0, sizeof(VelCapUniforms)),
            makeStorageBGLE(1, wgpu::BufferBindingType::Storage),
        };
        bgl_velcap_ = makeBGL(entries, "velcap_BGL");
        layout_velcap_ = makePipelineLayout(*bgl_velcap_);
        pipeline_velcap_ = makePipeline(*layout_velcap_, "post_process_vel", "velcap_Pipeline");
    }
}

void GpuStepOps::prepare(const GpuAtomBuffers& buffers) {
    const size_t vec4Bytes = buffers.countAtoms() * 4 * sizeof(float);

    auto makeBE = [](uint32_t binding, wgpu::Buffer buf, size_t size) {
        wgpu::BindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = size;
        return e;
    };

    // confine bindgroup
    {
        std::array<wgpu::BindGroupEntry, 3> entries{
            makeBE(0, sharedUniforms_, sizeof(ConfineUniforms)),
            makeBE(1, buffers.bufPos(), vec4Bytes),
            makeBE(2, buffers.bufVel(), vec4Bytes),
        };
        wgpu::BindGroupDescriptor d{};
        d.label = wgpu::StringView("confine_BindGroup");
        d.layout = *bgl_confine_;
        d.entryCount = entries.size();
        d.entries = entries.data();
        bindGroupConfine_ = WGPUContext::instance().device().createBindGroup(d);
    }

    // velcap bindgroup
    {
        std::array<wgpu::BindGroupEntry, 2> entries{
            makeBE(0, sharedUniforms_, sizeof(VelCapUniforms)),
            makeBE(1, buffers.bufVel(), vec4Bytes),
        };
        wgpu::BindGroupDescriptor d{};
        d.label = wgpu::StringView("velcap_BindGroup");
        d.layout = *bgl_velcap_;
        d.entryCount = entries.size();
        d.entries = entries.data();
        bindGroupVelCap_ = WGPUContext::instance().device().createBindGroup(d);
    }
}

void GpuStepOps::recordConfine(wgpu::CommandEncoder enc, uint32_t atomCount) {
    assert(isReady());
    assert(*bindGroupConfine_ != nullptr);

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("confine_to_box pass");
    wgpu::raii::ComputePassEncoder pass = enc.beginComputePass(pd);
    pass->setPipeline(*pipeline_confine_);

    uint32_t dynOffset = kConfineOffset;
    pass->setBindGroup(0, *bindGroupConfine_, 1, &dynOffset);
    pass->dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass->end();
}

void GpuStepOps::recordVelCap(wgpu::CommandEncoder enc, uint32_t atomCount) {
    assert(isReady());
    assert(*bindGroupVelCap_ != nullptr);

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("post_process_vel pass");
    wgpu::raii::ComputePassEncoder pass = enc.beginComputePass(pd);
    pass->setPipeline(*pipeline_velcap_);

    uint32_t dynOffset = kVelCapOffset;
    pass->setBindGroup(0, *bindGroupVelCap_, 1, &dynOffset);
    pass->dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass->end();
}
