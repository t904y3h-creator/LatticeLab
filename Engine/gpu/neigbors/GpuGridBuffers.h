#pragma once

#include <cstddef>
#include <string_view>

#include <webgpu/webgpu.hpp>

class SpatialGrid;

class GpuGridBuffers {
public:
    GpuGridBuffers() = default;

    GpuGridBuffers(const GpuGridBuffers&) = delete;
    GpuGridBuffers& operator=(const GpuGridBuffers&) = delete;

    // Пересоздаёт все буферы под новую ёмкость (в штуках атомов).
    void resize(size_t newAtoms, size_t newCountCells);

    // Download GPU → CPU (блокирующий)
    void downloadOffsets(SpatialGrid& s);
    void downloadAtomsInCells(SpatialGrid& s);
    void downloadCounts(SpatialGrid& s);

    wgpu::Buffer bufOffsets() const { return offsets; }
    wgpu::Buffer bufAtomsInCells() const { return atomsInCells; }
    wgpu::Buffer bufCount() const { return counts_; }
    wgpu::Buffer bufBlockSums() const { return blockSums_; }
    wgpu::Buffer bufReorderCursors() const { return reorderCursors; }

    size_t countAtoms() const { return countAtoms_; }
    size_t countCells() const { return countCells_; }

private:
    wgpu::Buffer makeStorageBuffer(size_t bytes, std::string_view label);

    // Блокирующее скачивание buf → tmpBuf_ (uintCount uint-ов).
    void downloadRaw(wgpu::Buffer src, size_t uintCount);

    size_t countAtoms_ = 0;
    size_t countCells_ = 0;

    wgpu::Buffer offsets;        // массив оффсетов (каждый оффсет - начало новой ячейки)
    wgpu::Buffer atomsInCells;   // атомы подряд сгруппированные по ячейкам
    wgpu::Buffer counts_;        // количество атомов в каждой ячейке
    wgpu::Buffer reorderCursors; // Вспомогательный счётчик для сортировки
    wgpu::Buffer blockSums_;     // ceil(countCells/1024) * u32

    std::vector<uint32_t> tmpBuf_;
};
