#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuAtomBuffers;

class GpuVerletPredict {
public:
    GpuVerletPredict();
    ~GpuVerletPredict() { release(); }

    GpuVerletPredict(const GpuVerletPredict&) = delete;
    GpuVerletPredict& operator=(const GpuVerletPredict&) = delete;

    // Должен быть вызван один раз после инициализации WebGPU.
    void init(wgpu::Device device, wgpu::Queue queue);
    void release();

    bool isReady() const { return pipeline_ != nullptr; }

    // Выполняет шейдер predict для atomCount атомов.
    // Буферы в buffers должны быть загружены перед вызовом:
    //   uploadPositions, uploadVelocities, uploadForces, uploadInvMass.
    // После вызова позиции в buffers обновлены; скачивайте downloadPositions.
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
