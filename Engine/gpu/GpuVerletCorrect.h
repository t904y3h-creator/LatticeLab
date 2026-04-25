#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

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

    bool isReady() const { return pipeline_ != nullptr; }

    void record(wgpu::CommandEncoder& enc, uint32_t atomCount);
    void prepare(const GpuAtomBuffers& buffers);

private:
    void buildPipeline();

    wgpu::Buffer sharedUniforms_;
    wgpu::BindGroup bindGroup_ = nullptr;
    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};