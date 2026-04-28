#pragma once
#include <cstdint>

#include <webgpu/webgpu-raii.hpp>

class GpuGridBuffers;
class World;

class GpuSpatialGrid {
public:
    struct Uniforms {
        float cellSize;
        uint32_t dx, dy, dz;
        uint32_t n;
        uint32_t pad;
    };
    static_assert(sizeof(Uniforms) == 24);
    static constexpr uint32_t kUniformOffset = 768;

    GpuSpatialGrid(wgpu::Buffer sharedUniforms) : sharedUniforms_(sharedUniforms) { buildPipelines(); }

    GpuSpatialGrid(const GpuSpatialGrid&) = delete;
    GpuSpatialGrid& operator=(const GpuSpatialGrid&) = delete;

    bool isReady() const { return *pipeline_count_ != nullptr; }

    void prepare(const GpuGridBuffers& gridBufs, wgpu::Buffer bufPos);
    void record(wgpu::CommandEncoder enc, const World& world);

private:
    void buildPipelines();

    wgpu::Buffer sharedUniforms_;
    wgpu::raii::BindGroup bindGroup_;

    wgpu::raii::ShaderModule shaderModule_;
    wgpu::raii::BindGroupLayout bindGroupLayout_;
    wgpu::raii::PipelineLayout pipelineLayout_;

    wgpu::raii::ComputePipeline pipeline_count_;
    wgpu::raii::ComputePipeline pipeline_scanBlocks_;
    wgpu::raii::ComputePipeline pipeline_scanLevel2_;
    wgpu::raii::ComputePipeline pipeline_addOffsets_;
    wgpu::raii::ComputePipeline pipeline_sort_;

    static constexpr uint32_t kWorkgroupCount = 64;
    static constexpr uint32_t kWorkgroupScan = 256;
};