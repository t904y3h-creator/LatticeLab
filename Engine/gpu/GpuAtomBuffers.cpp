#include "GpuAtomBuffers.h"

#include <cstring>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/WGPUContext.h"

wgpu::Buffer GpuAtomBuffers::makeStorageBuffer(size_t bytes, std::string_view label) {
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView(label);
    desc.size = bytes;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    desc.mappedAtCreation = false;
    return WGPUContext::instance().device().createBuffer(desc);
}

void GpuAtomBuffers::resize(size_t count) {
    countAtoms_ = count;

    const size_t vec4Bytes = count * 4 * sizeof(float);
    const size_t f32Bytes = count * sizeof(float);
    const size_t u32Bytes = count * sizeof(uint32_t);

    bufPos_ = makeStorageBuffer(vec4Bytes, "bufPos");
    bufVel_ = makeStorageBuffer(vec4Bytes, "bufVel");
    bufF_ = makeStorageBuffer(vec4Bytes, "bufF");
    bufPrevF_ = makeStorageBuffer(vec4Bytes, "bufPrevF");
    bufPe_ = makeStorageBuffer(f32Bytes, "bufPe");
    bufInvMass_ = makeStorageBuffer(f32Bytes, "bufInvMass");
    bufCharge_ = makeStorageBuffer(f32Bytes, "bufCharge");
    bufAtomType_ = makeStorageBuffer(u32Bytes, "bufAtomType");
    bufValence_ = makeStorageBuffer(u32Bytes, "bufValence");
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

void GpuAtomBuffers::downloadRaw(wgpu::Buffer src, size_t byteCount) {
    tmpBuf_.resize(byteCount);

    wgpu::BufferDescriptor desc{};
    desc.size = byteCount;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    desc.mappedAtCreation = false;
    wgpu::Buffer staging = WGPUContext::instance().device().createBuffer(desc);

    wgpu::CommandEncoder encoder = WGPUContext::instance().device().createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, staging, 0, byteCount);
    wgpu::CommandBuffer cmd = encoder.finish();
    WGPUContext::instance().queue().submit(1, &cmd);

    bool done = false;
    wgpu::BufferMapCallbackInfo callbackInfo{};
    callbackInfo.mode = wgpu::CallbackMode::AllowProcessEvents;
    callbackInfo.callback = [](WGPUMapAsyncStatus status, WGPUStringView label, void* userdata1, void*) {
        *reinterpret_cast<bool*>(userdata1) = true;
    };
    callbackInfo.userdata1 = &done;

    staging.mapAsync(wgpu::MapMode::Read, 0, byteCount, callbackInfo);

    while (!done) {
        WGPUContext::instance().processEvents();
    }

    const void* mapped = staging.getConstMappedRange(0, byteCount);
    std::memcpy(tmpBuf_.data(), mapped, byteCount);
    staging.unmap();
    staging.destroy();
}

void GpuAtomBuffers::downloadVec3(wgpu::Buffer buf, std::span<Vec3f> data) {
    downloadRaw(buf, data.size() * 4 * sizeof(float));

    const float* src = reinterpret_cast<const float*>(tmpBuf_.data());
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].x = src[i * 4 + 0];
        data[i].y = src[i * 4 + 1];
        data[i].z = src[i * 4 + 2];
    }
}

void GpuAtomBuffers::downloadFloat(wgpu::Buffer buf, std::span<float> data) {
    downloadRaw(buf, data.size_bytes());
    std::memcpy(data.data(), tmpBuf_.data(), data.size_bytes());
}

void GpuAtomBuffers::downloadU32(wgpu::Buffer buf, std::span<uint32_t> data) {
    downloadRaw(buf, data.size_bytes());
    std::memcpy(data.data(), tmpBuf_.data(), data.size_bytes());
}

// ── public upload ─────────────────────────────────────────────────────────────

void GpuAtomBuffers::uploadPositions(std::span<const Vec3f> v) { uploadVec3(bufPos_, v); }
void GpuAtomBuffers::uploadVelocities(std::span<const Vec3f> v) { uploadVec3(bufVel_, v); }
void GpuAtomBuffers::uploadForces(std::span<const Vec3f> v) { uploadVec3(bufF_, v); }
void GpuAtomBuffers::uploadPrevForces(std::span<const Vec3f> v) { uploadVec3(bufPrevF_, v); }
void GpuAtomBuffers::uploadPe(std::span<const float> v) { uploadFloat(bufPe_, v); }
void GpuAtomBuffers::uploadInvMass(std::span<const float> v) { uploadFloat(bufInvMass_, v); }
void GpuAtomBuffers::uploadCharge(std::span<const float> v) { uploadFloat(bufCharge_, v); }
void GpuAtomBuffers::uploadAtomType(std::span<const uint32_t> v) { uploadU32(bufAtomType_, v); }
void GpuAtomBuffers::uploadValence(std::span<const uint32_t> v) { uploadU32(bufValence_, v); }

// ── public download ───────────────────────────────────────────────────────────

void GpuAtomBuffers::downloadPositions(std::span<Vec3f> v) { downloadVec3(bufPos_, v); }
void GpuAtomBuffers::downloadVelocities(std::span<Vec3f> v) { downloadVec3(bufVel_, v); }
void GpuAtomBuffers::downloadForces(std::span<Vec3f> v) { downloadVec3(bufF_, v); }
void GpuAtomBuffers::downloadPrevForces(std::span<Vec3f> v) { downloadVec3(bufPrevF_, v); }
void GpuAtomBuffers::downloadPe(std::span<float> v) { downloadFloat(bufPe_, v); }
void GpuAtomBuffers::downloadInvMass(std::span<float> v) { downloadFloat(bufInvMass_, v); }
void GpuAtomBuffers::downloadCharge(std::span<float> v) { downloadFloat(bufCharge_, v); }
void GpuAtomBuffers::downloadAtomType(std::span<uint32_t> v) { downloadU32(bufAtomType_, v); }
void GpuAtomBuffers::downloadValence(std::span<uint32_t> v) { downloadU32(bufValence_, v); }