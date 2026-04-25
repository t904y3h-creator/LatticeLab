#include "GpuPhysicsPipeline.h"

#include <cassert>

#include <webgpu/webgpu.hpp>

#include "Engine/World.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/physics/ForceField.h"

void GpuPhysicsPipeline::init(const World& world) {
    const size_t n = atoms.size();
    const size_t cellCount = static_cast<size_t>((world.getGridSize().x - 2) * (world.getGridSize().y - 2) * (world.getGridSize().z - 2));

    // Буферы
    atomBufs_.resize(n);

    gridBufs_.resize(n, cellCount);

    // Инициализация
    gpuLJ_.init(forceField.ljForceField_, atoms);

    ready_ = true;
}

void GpuPhysicsPipeline::step(const World& world, const GpuStepParams& p) {
    assert(isReady());
    const uint32_t n = static_cast<uint32_t>(world.mobileCount());
    auto device = WGPUContext::instance().device();
    auto queue = WGPUContext::instance().queue();

    wgpu::CommandEncoder enc = device.createCommandEncoder({});

    gpuPredict_.record(enc, world.getAtomBuffers(), n, p.dt);
    gpuStepOps_.recordConfine(enc, atomBufs_, n, wallMax_.x, wallMax_.y, wallMax_.z);

    enc.copyBufferToBuffer(atomBufs_.bufF(), 0, atomBufs_.bufPrevF(), 0, n * sizeof(float) * 4);
    clearForcesAndEnergy(enc, n);

    // TODO сделать проверку на смещение cutoff
    // if (p.allowNLRebuild && neighborList.needsRebuild(atoms)) {
    gpuSpatialGrid_.record(enc, world);
    // }

    gpuWall_.record(enc, atomBufs_, n, wallMax_, gravity_);

    if (p.enableLJ) {
        // const float cutoffSqr = neighborList.cutoff() * neighborList.cutoff();
        gpuLJ_.record(enc, atomBufs_, gpuSpatialGrid_, n, world.getGridCellSize()); // cutoffSqr);
    }

    gpuCorrect_.record(enc, atomBufs_, n, p.dt, p.accelDamping);

    if (p.maxParticleSpeed > 0.0f) {
        gpuStepOps_.recordVelCap(enc, atomBufs_, n, p.maxParticleSpeed);
    }

    wgpu::CommandBuffer cmd = enc.finish({});
    queue.submit(1, &cmd);
}

void GpuPhysicsPipeline::clearForcesAndEnergy(wgpu::CommandEncoder enc, const World& world) {
    enc.clearBuffer(world.getAtomBuffers().bufF(), 0, atomCount * 4 * sizeof(float));
    enc.clearBuffer(world.getAtomBuffers().bufPe(), 0, atomCount * sizeof(float));
}
