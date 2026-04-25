#pragma once
#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuAtomBuffers;

class GpuStepOps {
public:
    struct ConfineUniforms {
        float maxX, maxY, maxZ;
        float restitution;
        uint32_t atomCount;
        uint32_t pad[3];
    };
    static_assert(sizeof(ConfineUniforms) == 32);
    static constexpr uint32_t kConfineOffset = 256;

    struct VelCapUniforms {
        float maxSpeedSqr;
        float maxSpeed;
        uint32_t atomCount;
        uint32_t pad;
    };
    static_assert(sizeof(VelCapUniforms) == 16);
    static constexpr uint32_t kVelCapOffset = 512;

    GpuStepOps(wgpu::Buffer sharedUniforms) : sharedUniforms_(sharedUniforms) { buildPipelines(); }

    GpuStepOps(const GpuStepOps&) = delete;
    GpuStepOps& operator=(const GpuStepOps&) = delete;

    bool isReady() const { return pipeline_confine_ != nullptr; }

    void prepare(const GpuAtomBuffers& buffers);
    void recordConfine(wgpu::CommandEncoder enc, uint32_t atomCount);
    void recordVelCap(wgpu::CommandEncoder enc, uint32_t atomCount);

private:
    void buildPipelines();

    wgpu::Buffer sharedUniforms_;

    wgpu::BindGroup bindGroupConfine_ = nullptr;
    wgpu::BindGroup bindGroupVelCap_ = nullptr;

    wgpu::ShaderModule shaderModule_ = nullptr;

    wgpu::BindGroupLayout bgl_confine_ = nullptr;
    wgpu::PipelineLayout layout_confine_ = nullptr;
    wgpu::ComputePipeline pipeline_confine_ = nullptr;

    wgpu::BindGroupLayout bgl_velcap_ = nullptr;
    wgpu::PipelineLayout layout_velcap_ = nullptr;
    wgpu::ComputePipeline pipeline_velcap_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};
