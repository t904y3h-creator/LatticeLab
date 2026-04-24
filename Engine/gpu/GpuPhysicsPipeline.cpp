#include "GpuPhysicsPipeline.h"

#include <cassert>

#include <webgpu/webgpu.hpp>

#include "Engine/NeighborSearch/NeighborList.h"
#include "Engine/SimBox.h"
#include "Engine/gpu/WGPUContext.h"
#include "Engine/physics/AtomStorage.h"
#include "Engine/physics/ForceField.h"

void GpuPhysicsPipeline::init(const AtomStorage& atoms, const ForceField& forceField, const World& box) {
    const size_t n = atoms.size();
    const size_t cellCount = static_cast<size_t>((box.grid.sizeX - 2) * (box.grid.sizeY - 2) * (box.grid.sizeZ - 2));

    // Буферы
    atomBufs_.resize(n);

    gridBufs_.resize(n, cellCount);

    // Инициализация
    gpuLJ_.init(forceField.ljForceField_, atoms);

    // Начальная загрузка статических данных
    atomBufs_.uploadInvMass(atoms, n);
    atomBufs_.uploadPositions(atoms, n);
    atomBufs_.uploadForces(atoms, n);
    atomBufs_.uploadPrevForces(atoms, n);
    atomBufs_.uploadVelocities(atoms, n);

    onBoxChanged(box);

    ready_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────

void GpuPhysicsPipeline::onAtomsChanged(const AtomStorage& atoms) {
    const size_t n = atoms.size();
    atomBufs_.resize(n);
    atomBufs_.uploadInvMass(atoms, n);
    atomBufs_.uploadPositions(atoms, n);
    atomBufs_.uploadForces(atoms, n);
    atomBufs_.uploadPrevForces(atoms, n);
    atomBufs_.uploadVelocities(atoms, n);
}

void GpuPhysicsPipeline::onBoxChanged(const World& box) {
    wallMax_ = box.size - Vec3f(1.0f, 1.0f, 1.0f);

    const size_t cellCount = static_cast<size_t>((box.grid.sizeX - 2) * (box.grid.sizeY - 2) * (box.grid.sizeZ - 2));
    gridBufs_.resize(atomBufs_.countAtoms(), cellCount);
}

void GpuPhysicsPipeline::step(AtomStorage& atoms, NeighborList& neighborList, const World& box, const GpuStepParams& p) {
    assert(isReady());
    const uint32_t n = static_cast<uint32_t>(atoms.mobileCount());
    auto device = WGPUContext::instance().device();
    auto queue = WGPUContext::instance().queue();

    wgpu::CommandEncoder enc = device.createCommandEncoder({});

    gpuPredict_.record(enc, atomBufs_, n, p.dt);
    gpuStepOps_.recordConfine(enc, atomBufs_, n, wallMax_.x, wallMax_.y, wallMax_.z);

    enc.copyBufferToBuffer(atomBufs_.bufF(), 0, atomBufs_.bufPrevF(), 0, n * sizeof(float) * 4);
    clearForcesAndEnergy(enc, n);

    // ── 4. Rebuild neighbor list (CPU) ────────────────────────────────────────
    if (p.allowNLRebuild && neighborList.needsRebuild(atoms)) {
        rebuildNeighborList(atoms, neighborList, box, n);
    }

    gpuWall_.record(enc, atomBufs_, n, wallMax_, gravity_);

    if (p.enableLJ) {
        const float cutoffSqr = neighborList.cutoff() * neighborList.cutoff();
        gpuLJ_.record(enc, atomBufs_, gpuSpatialGrid_, n, cutoffSqr);
    }

    gpuCorrect_.record(enc, atomBufs_, n, p.dt, p.accelDamping);

    if (p.maxParticleSpeed > 0.0f) {
        gpuStepOps_.recordVelCap(enc, atomBufs_, n, p.maxParticleSpeed);
    }

    wgpu::CommandBuffer cmd = enc.finish({});
    queue.submit(1, &cmd);
}

void GpuPhysicsPipeline::clearForcesAndEnergy(wgpu::CommandEncoder enc, uint32_t atomCount) {
    enc.clearBuffer(atomBufs_.bufF(), 0, atomCount * 4 * sizeof(float));
    enc.clearBuffer(atomBufs_.bufEnergy(), 0, atomCount * sizeof(float));
}

void GpuPhysicsPipeline::rebuildNeighborList(const AtomStorage& atoms, NeighborList& neighborList, const World& box, uint32_t atomCount) {
    // 1. CPU перестраивает свой список (нужен для needsRebuild в следующий раз)
    neighborList.build(atoms, const_cast<World&>(box));

    // 2. GPU перестраивает сетку (использует bufPos)
    gpuSpatialGrid_.dispatch(gridBufs_, atomBufs_.bufPos(), atomCount, box.grid);
}

// ─────────────────────────────────────────────────────────────────────────────

void GpuPhysicsPipeline::downloadState(AtomStorage& atoms) {
    const uint32_t n = static_cast<uint32_t>(atoms.size());
    atomBufs_.downloadPositions(atoms, n);
    atomBufs_.downloadVelocities(atoms, n);
}
