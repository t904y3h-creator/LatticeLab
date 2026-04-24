#include "GpuGridBuffers.h"

#include <cstdint>
#include <cstring>

#include "Engine/gpu/WGPUContext.h"

void GpuGridBuffers::resize(size_t newAtoms, size_t newCountCells) {
    if (countAtoms_ == newAtoms && countCells_ == newCountCells) {
        return;
    }

    countAtoms_ = newAtoms;
    countCells_ = newCountCells;

    const size_t blockCount = (newCountCells + 255) / 256; // TODO убрать захардкоженные значения

    auto& ctx = WGPUContext::instance();
    const auto storage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;

    offsets_ = ctx.createBuffer((countCells_ + 1) * sizeof(uint32_t), storage, "GridOffsets");
    counts_ = ctx.createBuffer(countCells_ * sizeof(uint32_t), storage, "GridCounts");
    atomsInCells_ = ctx.createBuffer(countAtoms_ * sizeof(uint32_t), storage, "GridAtomsInCells");
    reorderCursors_ = ctx.createBuffer((countCells_ + 1) * sizeof(uint32_t), storage, "GridReorderCursors");
    blockSums_ = ctx.createBuffer(blockCount * sizeof(uint32_t), storage, "GridBlockSums");
}
