#include "GpuGridBuffers.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "Engine/NeighborSearch/SpatialGrid.h"
#include "Engine/gpu/WGPUContext.h"

void GpuGridBuffers::resize(size_t newAtoms, size_t newCountCells) {
    if (countAtoms_ == newAtoms && countCells_ == newCountCells) {
        return;
    }

    countAtoms_ = newAtoms;
    countCells_ = newCountCells;

    offsets = makeStorageBuffer((countCells_ + 1) * sizeof(uint32_t), "GridOffsets");
    counts_ = makeStorageBuffer(countCells_ * sizeof(uint32_t), "GridCounts");
    atomsInCells = makeStorageBuffer(countAtoms_ * sizeof(uint32_t), "AtomsInCells");
    reorderCursors = makeStorageBuffer((countCells_ + 1) * sizeof(uint32_t), "GridReorderCursors");

    const size_t blockCount = (newCountCells + 255) / 256;
    blockSums_ = makeStorageBuffer(blockCount * sizeof(uint32_t), "GridBlockSums");
}

wgpu::Buffer GpuGridBuffers::makeStorageBuffer(size_t bytes, std::string_view label) {
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView(label);
    desc.size = bytes;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    desc.mappedAtCreation = false;
    return WGPUContext::instance().device().createBuffer(desc);
}
