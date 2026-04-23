#include "GpuAtomBuffers.h"

#include <cassert>
#include <cstring>

#include <webgpu.h>
#include <webgpu/webgpu.hpp>

#include "Engine/gpu/WGPUContext.h"
#include "Engine/physics/AtomStorage.h"

void GpuAtomBuffers::resize(size_t capacity) {
    if (capacity == capacity_) {
        return;
    }

    capacity_ = capacity;

    const size_t vec4Bytes = capacity * 4 * sizeof(float);
    const size_t f32Bytes = capacity * sizeof(float);

    bufPos_ = makeStorageBuffer(vec4Bytes, "AtomPos");
    bufVel_ = makeStorageBuffer(vec4Bytes, "AtomVel");
    bufF_ = makeStorageBuffer(vec4Bytes, "AtomForce");
    bufPrevF_ = makeStorageBuffer(vec4Bytes, "AtomPrevForce");
    bufInvMass_ = makeStorageBuffer(f32Bytes, "AtomInvMass");
}

wgpu::Buffer GpuAtomBuffers::makeStorageBuffer(size_t bytes, std::string_view label) {
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView(label);
    desc.size = bytes;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    desc.mappedAtCreation = false;
    return WGPUContext::instance().device().createBuffer(desc);
}

void GpuAtomBuffers::uploadVec4(wgpu::Buffer buf, const float* xs, const float* ys, const float* zs, size_t count) {
    tmpBuf_.resize(count * 4);
    for (size_t i = 0; i < count; ++i) {
        tmpBuf_[i * 4 + 0] = xs[i];
        tmpBuf_[i * 4 + 1] = ys[i];
        tmpBuf_[i * 4 + 2] = zs[i];
        tmpBuf_[i * 4 + 3] = 0.0f;
    }
    WGPUContext::instance().queue().writeBuffer(buf, 0, tmpBuf_.data(), count * 4 * sizeof(float));
}

void GpuAtomBuffers::uploadPositions(const AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    uploadVec4(bufPos_, s.xData(), s.yData(), s.zData(), count);
}

void GpuAtomBuffers::uploadVelocities(const AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    uploadVec4(bufVel_, s.vxData(), s.vyData(), s.vzData(), count);
}

void GpuAtomBuffers::uploadForces(const AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    uploadVec4(bufF_, s.fxData(), s.fyData(), s.fzData(), count);
}

void GpuAtomBuffers::uploadPrevForces(const AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    uploadVec4(bufPrevF_, s.pfxData(), s.pfyData(), s.pfzData(), count);
}

void GpuAtomBuffers::uploadInvMass(const AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    WGPUContext::instance().queue().writeBuffer(bufInvMass_, 0, s.invMassData(), count * sizeof(float));
}

void GpuAtomBuffers::downloadRaw(wgpu::Buffer src, size_t floatCount) {
    const size_t bytes = floatCount * sizeof(float);

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

    struct MapCtx {
        bool done = false;
    };
    MapCtx ctx;

    wgpu::BufferMapCallbackInfo callbackInfo{};
    callbackInfo.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) {
        static_cast<MapCtx*>(userdata1)->done = true;
    };
    callbackInfo.userdata1 = &ctx;
    staging.mapAsync(wgpu::MapMode::Read, 0, bytes, callbackInfo);

    while (!ctx.done) {
        WGPUContext::instance().device().poll(false, nullptr);
    }

    tmpBuf_.resize(floatCount);
    const void* mapped = staging.getConstMappedRange(0, bytes);
    std::memcpy(tmpBuf_.data(), mapped, bytes);

    staging.unmap();
    staging.destroy();
}

void GpuAtomBuffers::unpackVec4(float* xs, float* ys, float* zs, size_t count) {
    // tmpBuf_ должен быть заполнен через downloadRaw перед вызовом.
    for (size_t i = 0; i < count; ++i) {
        xs[i] = tmpBuf_[i * 4 + 0];
        ys[i] = tmpBuf_[i * 4 + 1];
        zs[i] = tmpBuf_[i * 4 + 2];
    }
}

void GpuAtomBuffers::downloadPositions(AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    downloadRaw(bufPos_, count * 4);
    unpackVec4(s.xData(), s.yData(), s.zData(), count);
}

void GpuAtomBuffers::downloadVelocities(AtomStorage& s, size_t count) {
    assert(count <= capacity_);
    downloadRaw(bufVel_, count * 4);
    unpackVec4(s.vxData(), s.vyData(), s.vzData(), count);
}