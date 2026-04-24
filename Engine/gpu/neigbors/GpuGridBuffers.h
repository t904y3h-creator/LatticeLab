#pragma once

#include <string_view>

#include <webgpu/webgpu.hpp>

class GpuGridBuffers {
public:
    GpuGridBuffers() = default;

    GpuGridBuffers(const GpuGridBuffers&) = delete;
    GpuGridBuffers& operator=(const GpuGridBuffers&) = delete;

    void resize(size_t newAtoms, size_t newCountCells);

    wgpu::Buffer bufOffsets() const { return offsets_; }
    wgpu::Buffer bufAtomsInCells() const { return atomsInCells_; }
    wgpu::Buffer bufCount() const { return counts_; }
    wgpu::Buffer bufBlockSums() const { return blockSums_; }
    wgpu::Buffer bufReorderCursors() const { return reorderCursors_; }

    size_t countAtoms() const { return countAtoms_; }
    size_t countCells() const { return countCells_; }

private:
    size_t countAtoms_ = 0;
    size_t countCells_ = 0;

    wgpu::Buffer offsets_;        // массив оффсетов (каждый оффсет - начало новой ячейки)
    wgpu::Buffer atomsInCells_;   // атомы подряд сгруппированные по ячейкам
    wgpu::Buffer counts_;         // количество атомов в каждой ячейке
    wgpu::Buffer reorderCursors_; // Вспомогательный счётчик для сортировки
    wgpu::Buffer blockSums_;      // ceil(countCells/1024) * u32
};
