#include "GpuPhysicsPipeline.h"

#include <cassert>

#include <webgpu/webgpu-raii.hpp>

#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"

GpuPhysicsPipeline::GpuPhysicsPipeline(const World& world)
    : sharedUniforms_(createSharedUniformBuffer()), gpuPredict_(sharedUniforms_), gpuCorrect_(sharedUniforms_),
      gpuStepOps_(sharedUniforms_), gpuSpatialGrid_(sharedUniforms_), gpuWall_(sharedUniforms_),
      gpuLJ_(world.getLJTable(), sharedUniforms_) {
    ready_ = true;

    prepare(world);
}

wgpu::Buffer GpuPhysicsPipeline::createSharedUniformBuffer() {
    wgpu::BufferDescriptor desc{};
    desc.label = wgpu::StringView("PhysicsSharedUniforms");
    desc.size = 1792;
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    return WGPUContext::instance().device().createBuffer(desc);
}

void GpuPhysicsPipeline::prepare(const World& world) {
    const auto& bufs = world.getAtomBuffers();
    gpuPredict_.prepare(bufs);
    gpuCorrect_.prepare(bufs);
    gpuStepOps_.prepare(bufs);
    gpuWall_.prepare(bufs);
    gpuLJ_.prepare(bufs, world.getGridBuffers());
    gpuSpatialGrid_.prepare(world.getGridBuffers(), bufs.bufPos());
}

void GpuPhysicsPipeline::step(wgpu::CommandEncoder enc, const World& world, const GpuStepParams& p) {
    assert(isReady());
    const uint32_t n = static_cast<uint32_t>(world.mobileCount());
    if (n == 0) {
        return;
    }

    struct UploadBlock {
        alignas(256) GpuVerletPredict::Uniforms verletPredict;
        alignas(256) GpuStepOps::ConfineUniforms confine;
        alignas(256) GpuStepOps::VelCapUniforms velCap;
        alignas(256) GpuSpatialGrid::Uniforms grid;
        alignas(256) GpuWallForceField::Uniforms wall;
        alignas(256) GpuLJForceField::Uniforms lj;
        alignas(256) GpuVerletCorrect::Uniforms verletCorrect;
    };
    static_assert(sizeof(UploadBlock) == 1792);
    const Vec3f& ws = world.getWorldSize();
    const Vec3f& gr = world.getGravity();
    float gridCellSzie = world.getGridCellSize();
    Vec3u gridSize = world.getGridSize();

    UploadBlock upload{};
    upload.verletPredict = {p.dt, n, {}};
    upload.confine = {ws.x, ws.y, ws.z, /*world.getRestitution()*/ 0.8, n, {}};
    upload.velCap = {p.maxParticleSpeed * p.maxParticleSpeed, p.maxParticleSpeed, n, 0};
    upload.grid = {gridCellSzie, gridSize.x, gridSize.y, gridSize.z, n, 0};
    upload.wall = {ws.x, ws.y, ws.z, gr.x, gr.y, gr.z, n, 0};
    upload.lj = {n, static_cast<uint32_t>(AtomData::Type::COUNT), gridSize.x, gridSize.y, gridSize.z, gridCellSzie, {}};
    upload.verletCorrect = {p.dt, p.accelDamping, n, 0};

    WGPUContext::instance().queue().writeBuffer(sharedUniforms_, 0, &upload, sizeof(upload));

    // 1. predict — обновить позиции
    gpuPredict_.record(enc, n);

    // 2. confine — отразить от стенок
    gpuStepOps_.recordConfine(enc, n);

    // 3. сохранить prevF, обнулить силы и энергию
    const size_t vec4Bytes = n * 4 * sizeof(float);
    enc.copyBufferToBuffer(world.getAtomBuffers().bufF(), 0, world.getAtomBuffers().bufPrevF(), 0, vec4Bytes);
    clearForcesAndEnergy(enc, world);

    // 4. перестройка сетки
    gpuSpatialGrid_.record(enc, world);

    // 5. wall forces + гравитация
    gpuWall_.record(enc, n);

    // 6. LJ силы
    if (world.isLJEnabled()) {
        gpuLJ_.record(enc, n);
    }

    // 7. correct — обновить скорости
    gpuCorrect_.record(enc, n);

    // 8. velCap — ограничить скорость
    if (p.maxParticleSpeed > 0.f) {
        gpuStepOps_.recordVelCap(enc, n);
    }
}

void GpuPhysicsPipeline::clearForcesAndEnergy(wgpu::CommandEncoder enc, const World& world) {
    const size_t n = world.mobileCount();
    enc.clearBuffer(world.getAtomBuffers().bufF(), 0, n * 4 * sizeof(float));
    enc.clearBuffer(world.getAtomBuffers().bufPe(), 0, n * sizeof(float));
}
