#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuAtomBuffers;

class GpuVerletPredict {
public:
    GpuVerletPredict();

    GpuVerletPredict(const GpuVerletPredict&) = delete;
    GpuVerletPredict& operator=(const GpuVerletPredict&) = delete;

    bool isReady() const { return pipeline_ != nullptr; }

    void dispatch(GpuAtomBuffers& buffers, uint32_t atomCount, float dt);

private:
    void buildPipeline();

    wgpu::BindGroup makeBindGroup(GpuAtomBuffers& buffers) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;
    wgpu::Buffer uniformBuffer_ = nullptr;

    // Количество потоков в одном workgroup (должно совпадать с шейдером).
    static constexpr uint32_t kWorkgroupSize = 64;
};
