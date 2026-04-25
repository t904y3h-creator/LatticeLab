#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuGridBuffers;
class World;

class GpuSpatialGrid {
public:
    GpuSpatialGrid();

    GpuSpatialGrid(const GpuSpatialGrid&) = delete;
    GpuSpatialGrid& operator=(const GpuSpatialGrid&) = delete;

    bool isReady() const { return pipeline_count_ != nullptr; }

    void record(wgpu::CommandEncoder enc, const World& world);

private:
    void buildPipelines();

    wgpu::BindGroup makeBindGroup(const GpuGridBuffers& gridBufs, wgpu::Buffer bufPos) const;

    void runPass(wgpu::CommandEncoder enc, wgpu::ComputePipeline pipeline, wgpu::BindGroup bindGroup, uint32_t workgroupsX,
                 std::string_view label) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;

    wgpu::ComputePipeline pipeline_count_ = nullptr;
    wgpu::ComputePipeline pipeline_scanBlocks_ = nullptr;
    wgpu::ComputePipeline pipeline_scanLevel2_ = nullptr;
    wgpu::ComputePipeline pipeline_addOffsets_ = nullptr;
    wgpu::ComputePipeline pipeline_sort_ = nullptr;

    wgpu::Buffer uniformBuffer_ = nullptr;

    static constexpr uint32_t kWorkgroupCount = 64;
    static constexpr uint32_t kWorkgroupScan = 256;
};
