#include "GpuWallForceField.h"

#include <array>
#include <cassert>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/WGPUContext.h"
#include "generated/shaders/force_wall.wgsl.h"

struct WallUniforms {
    float maxX, maxY, maxZ;
    float gravityX, gravityY, gravityZ;
    uint32_t atomCount;
    uint32_t _pad;
};
static_assert(sizeof(WallUniforms) == 32);

GpuWallForceField::GpuWallForceField() { buildPipeline(); }

void GpuWallForceField::buildPipeline() {
    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = wgpu::StringView(force_wallWGSL);

    wgpu::ShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    shaderModule_ = WGPUContext::instance().device().createShaderModule(smDesc);

    std::array<wgpu::BindGroupLayoutEntry, 3> bglEntries{};

    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Compute;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = sizeof(WallUniforms);

    auto makeStorageBGLE = [](uint32_t binding, wgpu::BufferBindingType type) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = wgpu::ShaderStage::Compute;
        e.buffer.type = type;
        e.buffer.hasDynamicOffset = false;
        e.buffer.minBindingSize = 0;
        return e;
    };

    bglEntries[1] = makeStorageBGLE(1, wgpu::BufferBindingType::ReadOnlyStorage);
    bglEntries[2] = makeStorageBGLE(2, wgpu::BufferBindingType::Storage);

    wgpu::BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    bindGroupLayout_ = WGPUContext::instance().device().createBindGroupLayout(bglDesc);

    WGPUBindGroupLayout rawBGL = bindGroupLayout_;
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &rawBGL;
    pipelineLayout_ = WGPUContext::instance().device().createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pipelineLayout_;
    cpDesc.compute.module = shaderModule_;
    cpDesc.compute.entryPoint = wgpu::StringView("main");
    pipeline_ = WGPUContext::instance().device().createComputePipeline(cpDesc);

    wgpu::BufferDescriptor ubDesc{};
    ubDesc.label = wgpu::StringView("Wall_Uniforms");
    ubDesc.size = sizeof(WallUniforms);
    ubDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    ubDesc.mappedAtCreation = false;
    uniformBuffer_ = WGPUContext::instance().device().createBuffer(ubDesc);
}

wgpu::BindGroup GpuWallForceField::makeBindGroup(const GpuAtomBuffers& atomBufs) const {
    const size_t vec4Bytes = atomBufs.countAtoms() * 4 * sizeof(float);

    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer_;
    entries[0].offset = 0;
    entries[0].size = sizeof(WallUniforms);

    entries[1].binding = 1;
    entries[1].buffer = atomBufs.bufPos();
    entries[1].offset = 0;
    entries[1].size = vec4Bytes;

    entries[2].binding = 2;
    entries[2].buffer = atomBufs.bufF();
    entries[2].offset = 0;
    entries[2].size = vec4Bytes;

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = static_cast<uint32_t>(entries.size());
    bgDesc.entries = entries.data();
    return WGPUContext::instance().device().createBindGroup(bgDesc);
}

void GpuWallForceField::record(wgpu::CommandEncoder enc, const GpuAtomBuffers& atomBufs, uint32_t atomCount, Vec3f wallMax, Vec3f gravity) {
    assert(isReady());

    WallUniforms uni{wallMax.x, wallMax.y, wallMax.z, gravity.x, gravity.y, gravity.z, atomCount, 0u};
    WGPUContext::instance().queue().writeBuffer(uniformBuffer_, 0, &uni, sizeof(uni));

    wgpu::BindGroup bg = makeBindGroup(atomBufs);

    wgpu::ComputePassDescriptor pd{};
    pd.label = wgpu::StringView("WallForces pass");
    auto pass = enc.beginComputePass(pd);
    pass.setPipeline(pipeline_);
    pass.setBindGroup(0, bg, 0, nullptr);
    pass.dispatchWorkgroups((atomCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
    pass.end();
}
