// GpuVerletPredict.h
#pragma once
#include <cstdint>

#include <webgpu/webgpu-raii.hpp>

class GpuAtomBuffers;

class GpuVerletPredict {
public:
    struct Uniforms {
        float dt;
        uint32_t atomCount;
        uint32_t pad[2];
    };
    static_assert(sizeof(Uniforms) == 16);
    static constexpr uint32_t kUniformOffset = 0;

    explicit GpuVerletPredict(wgpu::Buffer sharedUniforms) : sharedUniforms_(sharedUniforms) { buildPipeline(); }

    GpuVerletPredict(const GpuVerletPredict&) = delete;
    GpuVerletPredict& operator=(const GpuVerletPredict&) = delete;

    bool isReady() const { return *pipeline_ != nullptr; }

    // Вызвать один раз после создания GpuAtomBuffers
    void prepare(const GpuAtomBuffers& buffers);

    void record(wgpu::CommandEncoder& enc, uint32_t atomCount);

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