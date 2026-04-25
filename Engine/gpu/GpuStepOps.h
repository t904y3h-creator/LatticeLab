#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

#include "Engine/math/Vec3.h"

class GpuAtomBuffers;

class GpuStepOps {
public:
    GpuStepOps();

    GpuStepOps(const GpuStepOps&) = delete;
    GpuStepOps& operator=(const GpuStepOps&) = delete;

    bool isReady() const { return pipeline_confine_ != nullptr; }

    // Отражение от стенок. Обновляет pos и vel в buffers.
    void recordConfine(wgpu::CommandEncoder enc, const GpuAtomBuffers& buffers, uint32_t atomCount, Vec3f max);

    // Ограничение скорости. Обновляет vel в buffers.
    // Вызывать только если maxSpeed > 0.
    void recordVelCap(wgpu::CommandEncoder enc, const GpuAtomBuffers& buffers, uint32_t atomCount, float maxSpeed);

private:
    void buildPipelines();

    wgpu::BindGroup makeConfineBindGroup(const GpuAtomBuffers& buffers) const;
    wgpu::BindGroup makeVelCapBindGroup(const GpuAtomBuffers& buffers) const;

    wgpu::ShaderModule shaderModule_ = nullptr;

    // confine_to_box
    wgpu::BindGroupLayout bgl_confine_ = nullptr;
    wgpu::PipelineLayout layout_confine_ = nullptr;
    wgpu::ComputePipeline pipeline_confine_ = nullptr;
    wgpu::Buffer ub_confine_ = nullptr;

    // post_process_vel
    wgpu::BindGroupLayout bgl_velcap_ = nullptr;
    wgpu::PipelineLayout layout_velcap_ = nullptr;
    wgpu::ComputePipeline pipeline_velcap_ = nullptr;
    wgpu::Buffer ub_velcap_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};