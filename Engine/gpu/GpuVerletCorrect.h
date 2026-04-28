#pragma once

#include <cstdint>

#include <webgpu/webgpu-raii.hpp>

class GpuAtomBuffers;

class GpuVerletCorrect {
public:
    struct Uniforms {
        float dt;
        float accelDamping;
        uint32_t atomCount;
        uint32_t pad;
    };
    static_assert(sizeof(Uniforms) == 16);
    static constexpr uint32_t kUniformOffset = 1536;

    GpuVerletCorrect(wgpu::Buffer sharedUniforms) : sharedUniforms_(sharedUniforms) { buildPipeline(); }

    GpuVerletCorrect(const GpuVerletCorrect&) = delete;
    GpuVerletCorrect& operator=(const GpuVerletCorrect&) = delete;

    void init(wgpu::Device device, wgpu::Queue queue);

    bool isReady() const { return *pipeline_ != nullptr; }

    void record(wgpu::CommandEncoder& enc, uint32_t atomCount);
    void prepare(const GpuAtomBuffers& buffers);

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