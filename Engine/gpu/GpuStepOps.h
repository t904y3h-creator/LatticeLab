#pragma once
#include <cstdint>

#include <webgpu/webgpu-raii.hpp>

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

    bool isReady() const { return *pipeline_confine_ != nullptr; }

    void prepare(const GpuAtomBuffers& buffers);
    void recordConfine(wgpu::CommandEncoder enc, uint32_t atomCount);
    void recordVelCap(wgpu::CommandEncoder enc, uint32_t atomCount);

private:
    void buildPipelines();

    wgpu::Buffer sharedUniforms_;

    wgpu::raii::BindGroup bindGroupConfine_;
    wgpu::raii::BindGroup bindGroupVelCap_;

    wgpu::raii::ShaderModule shaderModule_;

    wgpu::raii::BindGroupLayout bgl_confine_;
    wgpu::raii::PipelineLayout layout_confine_;
    wgpu::raii::ComputePipeline pipeline_confine_;

    wgpu::raii::BindGroupLayout bgl_velcap_;
    wgpu::raii::PipelineLayout layout_velcap_;
    wgpu::raii::ComputePipeline pipeline_velcap_;

    static constexpr uint32_t kWorkgroupSize = 64;
};
