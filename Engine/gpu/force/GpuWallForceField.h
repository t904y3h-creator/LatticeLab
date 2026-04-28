#pragma once
#include <cstdint>

#include <webgpu/webgpu-raii.hpp>

class GpuAtomBuffers;

class GpuWallForceField {
public:
    struct Uniforms {
        float maxX, maxY, maxZ;
        float gravityX, gravityY, gravityZ;
        uint32_t atomCount;
        uint32_t pad;
    };
    static_assert(sizeof(Uniforms) == 32);
    static constexpr uint32_t kUniformOffset = 1024;

    GpuWallForceField(wgpu::Buffer sharedUniforms) : sharedUniforms_(sharedUniforms) { buildPipeline(); }

    GpuWallForceField(const GpuWallForceField&) = delete;
    GpuWallForceField& operator=(const GpuWallForceField&) = delete;

    bool isReady() const { return *pipeline_ != nullptr; }

    void prepare(const GpuAtomBuffers& atomBufs);
    void record(wgpu::CommandEncoder enc, uint32_t atomCount);

private:
    void buildPipeline();

    wgpu::Buffer sharedUniforms_;
    wgpu::raii::BindGroup bindGroup_;

    wgpu::raii::ShaderModule shaderModule_;
    wgpu::raii::BindGroupLayout bindGroupLayout_;
    wgpu::raii::PipelineLayout pipelineLayout_;
    wgpu::raii::ComputePipeline pipeline_;

    static constexpr uint32_t kWorkgroupSize = 64;
};
