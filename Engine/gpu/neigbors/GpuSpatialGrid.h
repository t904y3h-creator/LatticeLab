#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuGridBuffers;
class SpatialGrid;

// Выполняет построение пространственной сетки на GPU:
//   count → scan_blocks → scan_level2 → add_offsets → sort
//
// Все 5 проходов — в одном CommandEncoder, каждый в отдельном
// ComputePassEncoder (WebGPU требует границы между pass для
// гарантии видимости записей в storage буферах).

class GpuSpatialGrid {
public:
    GpuSpatialGrid() { buildPipelines(); };

    GpuSpatialGrid(const GpuSpatialGrid&) = delete;
    GpuSpatialGrid& operator=(const GpuSpatialGrid&) = delete;

    bool isReady() const { return pipeline_count_ != nullptr; }

    // Перестраивает сетку на GPU.
    // buffers должны быть resize() под текущий grid до вызова.
    // Позиции атомов уже загружены в GpuAtomBuffers::bufPos().
    void dispatch(GpuGridBuffers& gridBufs,
                  wgpu::Buffer bufPos, // vec4f позиции атомов
                  uint32_t atomCount,
                  const SpatialGrid& grid); // нужны dx,dy,dz,cellSize

private:
    void buildPipelines();

    // Пересоздаёт bind group под текущие буферы.
    // Все 5 проходов используют одну и ту же раскладку.
    wgpu::BindGroup makeBindGroup(GpuGridBuffers& gridBufs, wgpu::Buffer bufPos) const;

    void runPass(wgpu::CommandEncoder& enc, wgpu::ComputePipeline pipeline, wgpu::BindGroup bindGroup, uint32_t workgroupsX,
                 std::string_view label) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;

    // Отдельный pipeline на каждый из 5 проходов шейдера.
    wgpu::ComputePipeline pipeline_count_ = nullptr;
    wgpu::ComputePipeline pipeline_scanBlocks_ = nullptr;
    wgpu::ComputePipeline pipeline_scanLevel2_ = nullptr;
    wgpu::ComputePipeline pipeline_addOffsets_ = nullptr;
    wgpu::ComputePipeline pipeline_sort_ = nullptr;

    wgpu::Buffer uniformBuffer_ = nullptr;

    static constexpr uint32_t kWorkgroupCount = 64;
    static constexpr uint32_t kWorkgroupScan = 256;
};
