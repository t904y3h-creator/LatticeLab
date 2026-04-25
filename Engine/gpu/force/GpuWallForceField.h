#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

#include "Engine/math/Vec3.h"

class GpuAtomBuffers;

class GpuWallForceField {
public:
    GpuWallForceField();

    GpuWallForceField(const GpuWallForceField&) = delete;
    GpuWallForceField& operator=(const GpuWallForceField&) = delete;

    bool isReady() const { return pipeline_ != nullptr; }

    // wallMax = box.size - Vec3f(1,1,1)
    void record(wgpu::CommandEncoder enc, const GpuAtomBuffers& atomBufs, uint32_t atomCount, Vec3f wallMax, Vec3f gravity);

private:
    void buildPipeline();
    wgpu::BindGroup makeBindGroup(const GpuAtomBuffers& atomBufs) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;
    wgpu::Buffer uniformBuffer_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};
