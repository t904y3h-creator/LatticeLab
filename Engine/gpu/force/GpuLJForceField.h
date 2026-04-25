#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuAtomBuffers;
class GpuGridBuffers;
class LJTable;
class World;

class GpuLJForceField {
public:
    GpuLJForceField() = default;

    GpuLJForceField(const GpuLJForceField&) = delete;
    GpuLJForceField& operator=(const GpuLJForceField&) = delete;

    void init(const LJTable& ljTable);

    bool isReady() const { return pipeline_ != nullptr; }

    void record(wgpu::CommandEncoder enc, const World& world);

private:
    void buildPipeline();
    void uploadLJTable(const LJTable& ljTable);

    wgpu::BindGroup makeBindGroup(const GpuAtomBuffers& atomBufs, const GpuGridBuffers& gridBufs) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;
    wgpu::Buffer uniformBuffer_ = nullptr;
    wgpu::Buffer ljTableBuffer_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};
