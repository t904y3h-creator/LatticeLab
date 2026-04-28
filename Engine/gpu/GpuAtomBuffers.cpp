#include "GpuAtomBuffers.h"

#include <cstring>

#include <webgpu/webgpu-raii.hpp>

#include "Engine/gpu/WGPUContext.h"

void GpuAtomBuffers::resize(size_t count) {
    if (countAtoms_ == count) {
        return;
    }
    countAtoms_ = count;

    const size_t vec4Bytes = count * 4 * sizeof(float);
    const size_t f32Bytes = count * sizeof(float);
    const size_t u32Bytes = count * sizeof(uint32_t);

    WGPUContext& ctx = WGPUContext::instance();
    const auto storage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    bufPos_ = ctx.createBuffer(vec4Bytes, storage, "AtomPos");
    bufVel_ = ctx.createBuffer(vec4Bytes, storage, "AtomVel");
    bufF_ = ctx.createBuffer(vec4Bytes, storage, "AtomForce");
    bufPrevF_ = ctx.createBuffer(vec4Bytes, storage, "AtomPrevForce");
    bufPe_ = ctx.createBuffer(f32Bytes, storage, "AtomEnergy");
    bufInvMass_ = ctx.createBuffer(f32Bytes, storage, "AtomInvMass");
    bufCharge_ = ctx.createBuffer(f32Bytes, storage, "AtomCharge");
    bufAtomType_ = ctx.createBuffer(u32Bytes, storage, "AtomType");
    bufValence_ = ctx.createBuffer(u32Bytes, storage, "AtomValence");
}

void GpuAtomBuffers::uploadVec3(wgpu::Buffer buf, std::span<const Vec3f> data) {
    const size_t vec4Count = data.size();
    tmpBuf_.resize(vec4Count * 4 * sizeof(float));

    float* dst = reinterpret_cast<float*>(tmpBuf_.data());
    for (size_t i = 0; i < vec4Count; ++i) {
        dst[i * 4 + 0] = data[i].x;
        dst[i * 4 + 1] = data[i].y;
        dst[i * 4 + 2] = data[i].z;
        dst[i * 4 + 3] = 0.f;
    }

    WGPUContext::instance().queue().writeBuffer(buf, 0, tmpBuf_.data(), tmpBuf_.size());
}

void GpuAtomBuffers::uploadFloat(wgpu::Buffer buf, std::span<const float> data) {
    WGPUContext::instance().queue().writeBuffer(buf, 0, data.data(), data.size_bytes());
}

void GpuAtomBuffers::uploadU32(wgpu::Buffer buf, std::span<const uint32_t> data) {
    WGPUContext::instance().queue().writeBuffer(buf, 0, data.data(), data.size_bytes());
}

void GpuAtomBuffers::downloadRaw(wgpu::Buffer src, size_t byteCount) const {
    tmpBuf_.resize(byteCount);

    wgpu::BufferDescriptor desc{};
    desc.size = byteCount;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    desc.mappedAtCreation = false;
    wgpu::raii::Buffer staging = WGPUContext::instance().device().createBuffer(desc);

    wgpu::raii::CommandEncoder encoder = WGPUContext::instance().device().createCommandEncoder();
    encoder->copyBufferToBuffer(src, 0, *staging, 0, byteCount);
    wgpu::raii::CommandBuffer cmd = encoder->finish();
    WGPUContext::instance().queue().submit(1, &(*cmd));

    bool done = false;
    wgpu::BufferMapCallbackInfo callbackInfo{};
    callbackInfo.mode = wgpu::CallbackMode::AllowProcessEvents;
    callbackInfo.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* userdata1, void*) { *static_cast<bool*>(userdata1) = true; };
    callbackInfo.userdata1 = &done;

    staging->mapAsync(wgpu::MapMode::Read, 0, byteCount, callbackInfo);

    while (!done) {
        WGPUContext::instance().processEvents();
    }

    const void* mapped = staging->getConstMappedRange(0, byteCount);
    std::memcpy(tmpBuf_.data(), mapped, byteCount);
    staging->unmap();
    staging->destroy();
}

void GpuAtomBuffers::downloadVec3(wgpu::Buffer buf, std::span<Vec3f> data) const {
    downloadRaw(buf, data.size() * 4 * sizeof(float));

    const float* src = reinterpret_cast<const float*>(tmpBuf_.data());
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].x = src[i * 4 + 0];
        data[i].y = src[i * 4 + 1];
        data[i].z = src[i * 4 + 2];
    }
}

void GpuAtomBuffers::downloadFloat(wgpu::Buffer buf, std::span<float> data) const {
    downloadRaw(buf, data.size_bytes());
    std::memcpy(data.data(), tmpBuf_.data(), data.size_bytes());
}

void GpuAtomBuffers::downloadU32(wgpu::Buffer buf, std::span<uint32_t> data) const {
    downloadRaw(buf, data.size_bytes());
    std::memcpy(data.data(), tmpBuf_.data(), data.size_bytes());
}

void GpuAtomBuffers::uploadPositions(std::span<const Vec3f> v) { uploadVec3(bufPos_, v); }
void GpuAtomBuffers::uploadVelocities(std::span<const Vec3f> v) { uploadVec3(bufVel_, v); }
void GpuAtomBuffers::uploadForces(std::span<const Vec3f> v) { uploadVec3(bufF_, v); }
void GpuAtomBuffers::uploadPrevForces(std::span<const Vec3f> v) { uploadVec3(bufPrevF_, v); }
void GpuAtomBuffers::uploadPe(std::span<const float> v) { uploadFloat(bufPe_, v); }
void GpuAtomBuffers::uploadInvMass(std::span<const float> v) { uploadFloat(bufInvMass_, v); }
void GpuAtomBuffers::uploadCharge(std::span<const float> v) { uploadFloat(bufCharge_, v); }
void GpuAtomBuffers::uploadAtomType(std::span<const uint32_t> v) { uploadU32(bufAtomType_, v); }
void GpuAtomBuffers::uploadValence(std::span<const uint32_t> v) { uploadU32(bufValence_, v); }

void GpuAtomBuffers::downloadPositions(std::span<Vec3f> v) const { downloadVec3(bufPos_, v); }
void GpuAtomBuffers::downloadVelocities(std::span<Vec3f> v) const { downloadVec3(bufVel_, v); }
void GpuAtomBuffers::downloadForces(std::span<Vec3f> v) const { downloadVec3(bufF_, v); }
void GpuAtomBuffers::downloadPrevForces(std::span<Vec3f> v) const { downloadVec3(bufPrevF_, v); }
void GpuAtomBuffers::downloadPe(std::span<float> v) const { downloadFloat(bufPe_, v); }
void GpuAtomBuffers::downloadInvMass(std::span<float> v) const { downloadFloat(bufInvMass_, v); }
void GpuAtomBuffers::downloadCharge(std::span<float> v) const { downloadFloat(bufCharge_, v); }
void GpuAtomBuffers::downloadAtomType(std::span<uint32_t> v) const { downloadU32(bufAtomType_, v); }
void GpuAtomBuffers::downloadValence(std::span<uint32_t> v) const { downloadU32(bufValence_, v); }
