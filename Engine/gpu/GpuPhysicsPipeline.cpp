#include "GpuPhysicsPipeline.h"

#include <cassert>

#include <webgpu/webgpu.hpp>

#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"

GpuPhysicsPipeline::GpuPhysicsPipeline(const World& world) {
    gpuLJ_.init(world.getLJTable());
    ready_ = true;
}

void GpuPhysicsPipeline::step(const World& world, const GpuStepParams& p) {
    assert(isReady());
    const uint32_t n = static_cast<uint32_t>(world.mobileCount());
    if (n == 0) {
        return;
    }

    WGPUContext& ctx = WGPUContext::instance();
    wgpu::CommandEncoder enc = ctx.device().createCommandEncoder({});

    // 1. predict — обновить позиции
    gpuPredict_.record(enc, world.getAtomBuffers(), n, p.dt);

    // 2. confine — отразить от стенок
    gpuStepOps_.recordConfine(enc, world.getAtomBuffers(), n, world.getWorldSize());

    // 3. сохранить prevF, обнулить силы и энергию
    const size_t vec4Bytes = n * 4 * sizeof(float);
    enc.copyBufferToBuffer(world.getAtomBuffers().bufF(), 0, world.getAtomBuffers().bufPrevF(), 0, vec4Bytes);
    clearForcesAndEnergy(enc, world);

    // 4. перестройка сетки
    gpuSpatialGrid_.record(enc, world);

    // 5. wall forces + гравитация
    gpuWall_.record(enc, world.getAtomBuffers(), n, world.getWorldSize(), world.getGravity());

    // 6. LJ силы
    if (world.isLJEnabled()) {
        gpuLJ_.record(enc, world);
    }

    // 7. correct — обновить скорости
    gpuCorrect_.record(enc, world.getAtomBuffers(), n, p.dt, p.accelDamping);

    // 8. velCap — ограничить скорость
    if (p.maxParticleSpeed > 0.f) {
        gpuStepOps_.recordVelCap(enc, world.getAtomBuffers(), n, p.maxParticleSpeed);
    }

    wgpu::CommandBuffer cmd = enc.finish({});
    ctx.queue().submit(1, &cmd);
}

void GpuPhysicsPipeline::clearForcesAndEnergy(wgpu::CommandEncoder enc, const World& world) {
    const size_t n = world.mobileCount();
    enc.clearBuffer(world.getAtomBuffers().bufF(), 0, n * 4 * sizeof(float));
    enc.clearBuffer(world.getAtomBuffers().bufPe(), 0, n * sizeof(float));
}
