#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/neigbors/GpuSpatialGrid.h"

class GpuAtomBuffers;
class LJForceField;
class AtomStorage;

// GPU-реализация Lennard-Jones силового поля.
//
// Соответствует CPU-функции computePairInteractionsImpl<true, false>.
// Использует полный (full) neighbor list из GpuNeighborList.
//
// Перед dispatch должны быть актуальны на GPU:
//   GpuAtomBuffers: bufPos, bufF, bufEnergy
//   GpuNeighborList: bufOffsets, bufNeighbors
//   (загружаются через GpuAtomBuffers::upload* и GpuNeighborList::upload)
//
// После dispatch:
//   bufF и bufEnergy обновлены — скачивать не обязательно,
//   если дальнейшие шаги (correct) тоже на GPU.

class GpuLJForceField {
public:
    GpuLJForceField() = default;

    GpuLJForceField(const GpuLJForceField&) = delete;
    GpuLJForceField& operator=(const GpuLJForceField&) = delete;

    // ljForceField нужен только при init для построения LJ-таблицы.
    // После init можно не хранить ссылку.
    void init(const LJForceField& ljForceField, const AtomStorage& atoms);

    bool isReady() const { return pipeline_ != nullptr; }

    // Вычисляет LJ-силы и накапливает в bufF/bufEnergy.
    // cutoffSqr = cutoff * cutoff (из NeighborList::cutoff())
    void record(wgpu::CommandEncoder enc, GpuAtomBuffers& atomBufs, GpuSpatialGrid& gridBufs, uint32_t atomCount, float cutoffSqr);

private:
    void buildPipeline();
    void uploadLJTable(const LJForceField& ljForceField);
    void uploadAtomTypes(const AtomStorage& atoms);

    wgpu::BindGroup makeBindGroup(GpuAtomBuffers& atomBufs, GpuSpatialGrid& gridBufs) const;

    wgpu::ShaderModule shaderModule_ = nullptr;
    wgpu::BindGroupLayout bindGroupLayout_ = nullptr;
    wgpu::PipelineLayout pipelineLayout_ = nullptr;
    wgpu::ComputePipeline pipeline_ = nullptr;
    wgpu::Buffer uniformBuffer_ = nullptr;

    // Статические данные — загружаются один раз при init
    wgpu::Buffer ljTableBuffer_ = nullptr;   // TypeCount² × vec2f(C6, C12)
    wgpu::Buffer atomTypesBuffer_ = nullptr; // atomCount × u32

    uint32_t typeCount_ = 0; // AtomData::Type::COUNT

    static constexpr uint32_t kWorkgroupSize = 64;
};