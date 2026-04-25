#pragma once
#include <cstdint>

#include <webgpu/webgpu.hpp>

class GpuAtomBuffers;
class GpuGridBuffers;
class LJTable;
class World;

class GpuLJForceField {
public:
    struct Uniforms {
        uint32_t atomCount;
        uint32_t typeCount;
        uint32_t grid_dx;
        uint32_t grid_dy;
        uint32_t grid_dz;
        float grid_cell_size;
        uint32_t pad[2];
    };
    static_assert(sizeof(Uniforms) == 32);
    static constexpr uint32_t kUniformOffset = 1280;

    GpuLJForceField(const LJTable& ljTable, wgpu::Buffer sharedUniforms) : sharedUniforms_(sharedUniforms) {
        buildPipeline();
        uploadLJTable(ljTable);
    }

    GpuLJForceField(const GpuLJForceField&) = delete;
    GpuLJForceField& operator=(const GpuLJForceField&) = delete;

    bool isReady() const { return pipeline_ != nullptr; }

    void prepare(const GpuAtomBuffers& atomBufs, const GpuGridBuffers& gridBufs);
    void record(wgpu::CommandEncoder enc, uint32_t atomCount);

private:
    void buildPipeline();
    void uploadLJTable(const LJTable& ljTable);

    wgpu::Buffer sharedUniforms_;
    wgpu::BindGroup bindGroup_ = nullptr;
    wgpu::Buffer ljTableBuffer_ = nullptr;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;

    static constexpr uint32_t kWorkgroupSize = 64;
};
