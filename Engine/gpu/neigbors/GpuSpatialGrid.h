#pragma once
#include <cstdint>
#include <string_view>

#include <webgpu/webgpu.hpp>

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

    bool isReady() const { return pipeline_count_ != nullptr; }

    void prepare(const GpuGridBuffers& gridBufs, wgpu::Buffer bufPos);
    void record(wgpu::CommandEncoder enc, const World& world);

private:
    void buildPipelines();
    void runPass(wgpu::CommandEncoder enc, wgpu::ComputePipeline pipeline, uint32_t workgroupsX, std::string_view label) const;

    wgpu::Buffer sharedUniforms_;
    wgpu::BindGroup bindGroup_ = nullptr;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;

    wgpu::ComputePipeline pipeline_count_ = nullptr;
    wgpu::ComputePipeline pipeline_scanBlocks_ = nullptr;
    wgpu::ComputePipeline pipeline_scanLevel2_ = nullptr;
    wgpu::ComputePipeline pipeline_addOffsets_ = nullptr;
    wgpu::ComputePipeline pipeline_sort_ = nullptr;

    static constexpr uint32_t kWorkgroupCount = 64;
    static constexpr uint32_t kWorkgroupScan = 256;
};