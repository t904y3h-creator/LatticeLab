#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuAtomBuffers;

class GpuVerletCorrect {
public:
    GpuVerletCorrect();

    GpuVerletCorrect(const GpuVerletCorrect&) = delete;
    GpuVerletCorrect& operator=(const GpuVerletCorrect&) = delete;

    void init(wgpu::Device device, wgpu::Queue queue);

    bool isReady() const { return pipeline_ != nullptr; }

    // Выполняет шейдер correct для atomCount атомов.
    // Перед вызовом в buffers должны быть загружены:
    //   uploadVelocities, uploadForces, uploadPrevForces, uploadInvMass.
    // После вызова скорости обновлены; скачивайте downloadVelocities.
    void record(wgpu::CommandEncoder& enc, const GpuAtomBuffers& buffers, uint32_t atomCount, float dt, float accelDamping);

private:
    void buildPipeline();
    wgpu::BindGroup makeBindGroup(const GpuAtomBuffers& buffers) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;
    wgpu::Buffer uniformBuffer_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};