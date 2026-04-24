#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <webgpu/webgpu.hpp>

#include "Engine/math/Vec3.h"

class GpuAtomBuffers {
public:
    GpuAtomBuffers() = default;

    GpuAtomBuffers(const GpuAtomBuffers&) = delete;
    GpuAtomBuffers& operator=(const GpuAtomBuffers&) = delete;

    void resize(size_t newCountAtoms);

    size_t countAtoms() const { return countAtoms_; }

    // Upload CPU → GPU
    void uploadPositions(std::span<const Vec3f> v);
    void uploadVelocities(std::span<const Vec3f> v);
    void uploadForces(std::span<const Vec3f> v);
    void uploadPrevForces(std::span<const Vec3f> v);
    void uploadPe(std::span<const float> v);
    void uploadInvMass(std::span<const float> v);
    void uploadCharge(std::span<const float> v);
    void uploadAtomType(std::span<const uint32_t> v);
    void uploadValence(std::span<const uint32_t> v);

    // Download GPU → CPU
    void downloadPositions(std::span<Vec3f> v);
    void downloadVelocities(std::span<Vec3f> v);
    void downloadForces(std::span<Vec3f> v);
    void downloadPrevForces(std::span<Vec3f> v);
    void downloadPe(std::span<float> v);
    void downloadInvMass(std::span<float> v);
    void downloadCharge(std::span<float> v);
    void downloadAtomType(std::span<uint32_t> v);
    void downloadValence(std::span<uint32_t> v);

    wgpu::Buffer bufPos() const { return bufPos_; }
    wgpu::Buffer bufVel() const { return bufVel_; }
    wgpu::Buffer bufF() const { return bufF_; }
    wgpu::Buffer bufPrevF() const { return bufPrevF_; }
    wgpu::Buffer bufPe() const { return bufPe_; }
    wgpu::Buffer bufInvMass() const { return bufInvMass_; }
    wgpu::Buffer bufCharge() const { return bufCharge_; }
    wgpu::Buffer bufAtomType() const { return bufAtomType_; }
    wgpu::Buffer bufValence() const { return bufValence_; }

private:
    wgpu::Buffer makeStorageBuffer(size_t bytes, std::string_view label);

    void uploadVec3(wgpu::Buffer buf, std::span<const Vec3f> data);
    void uploadFloat(wgpu::Buffer buf, std::span<const float> data);
    void uploadU32(wgpu::Buffer buf, std::span<const uint32_t> data);

    void downloadRaw(wgpu::Buffer src, size_t byteCount);

    void downloadVec3(wgpu::Buffer buf, std::span<Vec3f> data);
    void downloadFloat(wgpu::Buffer buf, std::span<float> data);
    void downloadU32(wgpu::Buffer buf, std::span<uint32_t> data);

    size_t countAtoms_ = 0;

    wgpu::Buffer bufPos_ = nullptr;      // array<vec4f>  [x, y, z, 0]
    wgpu::Buffer bufVel_ = nullptr;      // array<vec4f>  [vx,vy,vz,0]
    wgpu::Buffer bufF_ = nullptr;        // array<vec4f>  [fx,fy,fz,0]
    wgpu::Buffer bufPrevF_ = nullptr;    // array<vec4f>  [pfx,pfy,pfz,0]
    wgpu::Buffer bufPe_ = nullptr;       // array<f32>
    wgpu::Buffer bufInvMass_ = nullptr;  // array<f32>
    wgpu::Buffer bufCharge_ = nullptr;   // array<f32>
    wgpu::Buffer bufAtomType_ = nullptr; // array<u32>
    wgpu::Buffer bufValence_ = nullptr;  // array<u32>

    std::vector<std::byte> tmpBuf_;
};