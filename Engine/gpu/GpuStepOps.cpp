#include "GpuStepOps.h"

#include <array>
#include <cassert>
#include <span>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/step_ops.wgsl.h"

struct ConfineUniforms {
    float maxX, maxY, maxZ;
    float restitution;
    uint32_t atomCount;
    uint32_t _pad[3];
};
static_assert(sizeof(ConfineUniforms) == 32);

struct VelCapUniforms {
    float maxSpeedSqr;
    float maxSpeed;
    uint32_t atomCount;
    uint32_t _pad;
};
static_assert(sizeof(VelCapUniforms) == 16);

GpuStepOps::GpuStepOps() { buildPipelines(); }

void GpuStepOps::buildPipelines() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(step_opsWGSL);
    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    auto makeUB = [&](size_t bytes, std::string_view label) -> wgpu::Buffer {
        wgpu::BufferDescriptor d{};
        d.label = wgpu::StringView(label);
        d.size = bytes;
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        d.mappedAtCreation = false;
        return WGPUContext::instance().device().createBuffer(d);
    };

    auto makeStorageBGLE = [](uint32_t binding, wgpu::BufferBindingType type) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        e.buffer.minBindingSize = 0;
        return e;
    };

    auto makeUniformBGLE = [](uint32_t binding, uint64_t minSize) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = wgpu::BufferBindingType::Uniform;
        e.buffer.hasDynamicOffset = false;
        e.buffer.minBindingSize = minSize;
        return e;
    };

    auto makeBGL = [&](std::span<wgpu::BindGroupLayoutEntry> entries, std::string_view label) -> wgpu::BindGroupLayout {
        wgpu::BindGroupLayoutDescriptor d{};
        d.label = wgpu::StringView(label);
        d.entryCount = static_cast<uint32_t>(entries.size());
        d.entries = entries.data();
        return WGPUContext::instance().device().createBindGroupLayout(d);
    };

    auto makePipelineLayout = [&](wgpu::BindGroupLayout bgl) -> wgpu::PipelineLayout {
        WGPUBindGroupLayout raw = bgl;
        wgpu::PipelineLayoutDescriptor d{};
        d.bindGroupLayoutCount = 1;
        d.bindGroupLayouts = &raw;
        return WGPUContext::instance().device().createPipelineLayout(d);
    };

    auto makePipeline = [&](wgpu::PipelineLayout layout, std::string_view entry, std::string_view label) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor d{};
        d.label = wgpu::StringView(label);
        d.layout = layout;
        d.compute.module = shaderModule_;
        d.compute.entryPoint = wgpu::StringView(entry);
        return WGPUContext::instance().device().createComputePipeline(d);
    };

    // confine_to_box: group(0) — uniform, pos (rw), vel (rw)
    {
        std::array<wgpu::BindGroupLayoutEntry, 3> entries{
            makeUniformBGLE(0, sizeof(ConfineUniforms)),
            makeStorageBGLE(1, wgpu::BufferBindingType::Storage),
            makeStorageBGLE(2, wgpu::BufferBindingType::Storage),
        };
        bgl_confine_ = makeBGL(entries, "confineToBox_BindGroupLayout");
        layout_confine_ = makePipelineLayout(bgl_confine_);
        pipeline_confine_ = makePipeline(layout_confine_, "confine_to_box", "confineToBox_Pipeline");
        ub_confine_ = makeUB(sizeof(ConfineUniforms), "ConfineUniforms");
    }

    // post_process_vel: group(1) — uniform, vel (rw)
    {
        std::array<wgpu::BindGroupLayoutEntry, 2> entries{
            makeUniformBGLE(0, sizeof(VelCapUniforms)),
            makeStorageBGLE(1, wgpu::BufferBindingType::Storage),
        };
        bgl_velcap_ = makeBGL(entries, "postProcessVel_BindGroupLayout");
        layout_velcap_ = makePipelineLayout(bgl_velcap_);
        pipeline_velcap_ = makePipeline(layout_velcap_, "post_process_vel", "postProcessVel_Pipeline");
        ub_velcap_ = makeUB(sizeof(VelCapUniforms), "VelCapUniforms");
    }
}

wgpu::BindGroup GpuStepOps::makeConfineBindGroup(GpuAtomBuffers& buffers) const {
    const size_t vec4Bytes = buffers.capacity() * 4 * sizeof(float);

    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0].binding = 0;
    entries[0].buffer = ub_confine_;
    entries[0].offset = 0;
    entries[0].size = sizeof(ConfineUniforms);

    entries[1].binding = 1;
    entries[1].buffer = buffers.bufPos();
    entries[1].offset = 0;
    entries[1].size = vec4Bytes;

    entries[2].binding = 2;
    entries[2].buffer = buffers.bufVel();
    entries[2].offset = 0;
    entries[2].size = vec4Bytes;

    wgpu::BindGroupDescriptor d{};
    d.layout = bgl_confine_;
    d.entryCount = static_cast<uint32_t>(entries.size());
    d.entries = entries.data();
    return WGPUContext::instance().device().createBindGroup(d);
}

wgpu::BindGroup GpuStepOps::makeVelCapBindGroup(GpuAtomBuffers& buffers) const {
    const size_t vec4Bytes = buffers.capacity() * 4 * sizeof(float);

    std::array<wgpu::BindGroupEntry, 2> entries{};
    entries[0].binding = 0;
    entries[0].buffer = ub_velcap_;
    entries[0].offset = 0;
    entries[0].size = sizeof(VelCapUniforms);

    entries[1].binding = 1;
    entries[1].buffer = buffers.bufVel();
    entries[1].offset = 0;
    entries[1].size = vec4Bytes;

    wgpu::BindGroupDescriptor d{};
    d.layout = bgl_velcap_;
    d.entryCount = static_cast<uint32_t>(entries.size());
    d.entries = entries.data();
    return WGPUContext::instance().device().createBindGroup(d);
}

void GpuStepOps::dispatchConfine(GpuAtomBuffers& buffers, uint32_t atomCount, float maxX, float maxY, float maxZ) {
    assert(isReady());

    ConfineUniforms uni{maxX, maxY, maxZ, 0.8f, atomCount, {0, 0, 0}};
    WGPUContext::instance().queue().writeBuffer(ub_confine_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bg = makeConfineBindGroup(buffers);
    wgpu::CommandEncoder enc = WGPUContext::instance().device().createCommandEncoder({});
    {
        wgpu::ComputePassDescriptor pd{};
        pd.label = wgpu::StringView("confine_to_box");
        auto pass = enc.beginComputePass(pd);
        pass.setPipeline(pipeline_confine_);
        pass.setBindGroup(0, bg, 0, nullptr);
        pass.dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        pass.end();
        pass.release();
    }
    wgpu::CommandBuffer cmd = enc.finish({});
    WGPUContext::instance().queue().submit(1, &cmd);
    WGPUContext::instance().processEvents();
}

void GpuStepOps::dispatchVelCap(GpuAtomBuffers& buffers, uint32_t atomCount, float maxSpeed) {
    assert(isReady());
    assert(maxSpeed > 0.0f);

    VelCapUniforms uni{maxSpeed * maxSpeed, maxSpeed, atomCount, 0u};
    WGPUContext::instance().queue().writeBuffer(ub_velcap_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bg = makeVelCapBindGroup(buffers);
    wgpu::CommandEncoder enc = WGPUContext::instance().device().createCommandEncoder({});
    {
        wgpu::ComputePassDescriptor pd{};
        pd.label = wgpu::StringView("post_process_vel");
        auto pass = enc.beginComputePass(pd);
        pass.setPipeline(pipeline_velcap_);
        pass.setBindGroup(0, bg, 0, nullptr);
        pass.dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        pass.end();
        pass.release();
    }
    wgpu::CommandBuffer cmd = enc.finish({});
    WGPUContext::instance().queue().submit(1, &cmd);
    WGPUContext::instance().processEvents();
}