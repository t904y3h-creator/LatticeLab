#pragma once

#include <cstddef>
#include <vector>

#include <webgpu/webgpu.hpp>

class AtomStorage;

class GpuAtomBuffers {
public:
    GpuAtomBuffers() = default;

    GpuAtomBuffers(const GpuAtomBuffers&) = delete;
    GpuAtomBuffers& operator=(const GpuAtomBuffers&) = delete;

    // Пересоздаёт все буферы под новую ёмкость (в штуках атомов).
    // Вызывать при изменении capacity в AtomStorage.
    void resize(size_t capacity);

    void release();

    size_t capacity() const { return capacity_; }

    // ── Upload CPU → GPU ────────────────────────────────────────────────────
    void uploadPositions(const AtomStorage& s, size_t count);
    void uploadVelocities(const AtomStorage& s, size_t count);
    void uploadForces(const AtomStorage& s, size_t count);
    void uploadPrevForces(const AtomStorage& s, size_t count);
    void uploadInvMass(const AtomStorage& s, size_t count);

    // ── Download GPU → CPU (блокирующий) ───────────────────────────────────
    void downloadPositions(AtomStorage& s, size_t count);
    void downloadVelocities(AtomStorage& s, size_t count);

    wgpu::Buffer bufPos() const { return bufPos_; }
    wgpu::Buffer bufVel() const { return bufVel_; }
    wgpu::Buffer bufF() const { return bufF_; }
    wgpu::Buffer bufPrevF() const { return bufPrevF_; }
    wgpu::Buffer bufInvMass() const { return bufInvMass_; }

private:
    wgpu::Buffer makeStorageBuffer(size_t bytes, const char* label);

    // Упаковка SoA (xs, ys, zs) → interleaved vec4 и запись в buf.
    void uploadVec4(wgpu::Buffer buf, const float* xs, const float* ys, const float* zs, size_t count);

    // Блокирующее скачивание buf → tmpBuf_ (floatCount float-ов).
    void downloadRaw(wgpu::Buffer src, size_t floatCount);

    // Распаковка tmpBuf_ (vec4) → SoA (xs, ys, zs).
    void unpackVec4(float* xs, float* ys, float* zs, size_t count);

    size_t capacity_ = 0;

    wgpu::Buffer bufPos_ = nullptr;     // : array<vec4f> - [x, y, z, 0]
    wgpu::Buffer bufVel_ = nullptr;     // : array<vec4f> - [vx,vy,vz,0]
    wgpu::Buffer bufF_ = nullptr;       // : array<vec4f> - [fx,fy,fz,0]
    wgpu::Buffer bufPrevF_ = nullptr;   // : array<vec4f> - [pfx,pfy,pfz,0]
    wgpu::Buffer bufInvMass_ = nullptr; // : array<f32>   - invMass

    std::vector<float> tmpBuf_;
};