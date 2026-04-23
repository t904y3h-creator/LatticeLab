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

void GpuGridBuffers::downloadOffsets(SpatialGrid& s) {
    downloadRaw(offsets, s.offsets().size());
    std::ranges::copy(tmpBuf_, s.offsets().begin());
}

void GpuGridBuffers::downloadAtomsInCells(SpatialGrid& s) {
    downloadRaw(atomsInCells, s.atomsInCells().size());
    std::ranges::copy(tmpBuf_, s.atomsInCells().begin());
}
void GpuGridBuffers::downloadCounts(SpatialGrid& s) {
    downloadRaw(counts_, s.counts().size());
    std::ranges::copy(tmpBuf_, s.counts().begin());
}

wgpu::Buffer GpuGridBuffers::makeStorageBuffer(size_t bytes, std::string_view label) {
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView(label);
    desc.size = bytes;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    desc.mappedAtCreation = false;
    return WGPUContext::instance().device().createBuffer(desc);
}

void GpuGridBuffers::downloadRaw(wgpu::Buffer src, size_t uintCount) {
    const size_t bytes = uintCount * sizeof(uint32_t);

    wgpu::BufferDescriptor stagingDesc{};
    stagingDesc.label = wgpu::StringView("staging_readback");
    stagingDesc.size = bytes;
    stagingDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    stagingDesc.mappedAtCreation = false;
    wgpu::Buffer staging = WGPUContext::instance().device().createBuffer(stagingDesc);

    {
        wgpu::CommandEncoder enc = WGPUContext::instance().device().createCommandEncoder({});
        enc.copyBufferToBuffer(src, 0, staging, 0, bytes);
        wgpu::CommandBuffer cmd = enc.finish({});
        WGPUContext::instance().queue().submit(1, &cmd);
    }

    wgpu::BufferMapCallbackInfo callbackInfo{};
    callbackInfo.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) { *static_cast<bool*>(userdata1) = true; };
    bool done = false;
    callbackInfo.userdata1 = &done;
    staging.mapAsync(wgpu::MapMode::Read, 0, bytes, callbackInfo);

    while (!done) {
        WGPUContext::instance().device().poll(false, nullptr);
    }

    tmpBuf_.resize(uintCount);
    const void* mapped = staging.getConstMappedRange(0, bytes);
    std::memcpy(tmpBuf_.data(), mapped, bytes);

    staging.unmap();
    staging.destroy();
}