#pragma once

#include <cstdint>

#include <webgpu/webgpu.hpp>

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/GpuStepOps.h"
#include "Engine/gpu/GpuVerletCorrect.h"
#include "Engine/gpu/GpuVerletPredict.h"
#include "Engine/gpu/force/GpuLJForceField.h"
#include "Engine/gpu/force/GpuWallForceField.h"
#include "Engine/gpu/neigbors/GpuGridBuffers.h"
#include "Engine/gpu/neigbors/GpuSpatialGrid.h"
#include "Engine/math/Vec3.h"

class AtomStorage;
class NeighborList;
class ForceField;
class World;

// Параметры одного шага — зеркало StepData, но без CPU-данных
struct GpuStepParams {
    float dt;
    float accelDamping;
    float maxParticleSpeed; // 0 = отключено
    bool enableLJ;
    bool allowNLRebuild;
};

// ─────────────────────────────────────────────────────────────────────────────
// GpuPhysicsPipeline
//
// Владеет всеми GPU-ресурсами и оркестрирует полный шаг симуляции на GPU.
//
// Порядок шага:
//   1. predict         — обновить позиции
//   2. confineToBox    — отразить от стенок
//   3. clearForces     — обнулить силы/энергию
//   4. [NL rebuild]    — если нужно: сетка → full NL
//   5. wallForces      — мягкие стены + гравитация
//   6. ljForces        — LJ попарные взаимодействия
//   7. correct         — обновить скорости
//   8. [velCap]        — ограничить максимальную скорость
//
// CPU остаётся ответственным за:
//   - Coulomb (пока не перенесён)
//   - Bond forces (пока не перенесены)
//   - needsRebuild проверку (читает CPU-позиции, обновляемые по требованию)
// ─────────────────────────────────────────────────────────────────────────────

class GpuPhysicsPipeline {
public:
    GpuPhysicsPipeline() = default;
    ~GpuPhysicsPipeline() = default;

    GpuPhysicsPipeline(const GpuPhysicsPipeline&) = delete;
    GpuPhysicsPipeline& operator=(const GpuPhysicsPipeline&) = delete;

    // Инициализация — вызвать один раз после создания wgpu::Device.
    void init(const AtomStorage& atoms, const ForceField& forceField, const World& box);

    bool isReady() const { return ready_; }

    // Вызывать при изменении состава атомов (добавление/удаление).
    void onAtomsChanged(const AtomStorage& atoms);

    // Вызывать при изменении размера бокса.
    void onBoxChanged(const World& box);

    // Установить гравитацию.
    void setGravity(Vec3f gravity) { gravity_ = gravity; }

    // Полный шаг симуляции на GPU.
    // CPU NeighborList нужен только для проверки needsRebuild и как
    // источник данных при rebuild — не модифицируется.
    void step(AtomStorage& atoms, NeighborList& neighborList, const World& box, const GpuStepParams& params);

    // Скачать позиции и скорости GPU → CPU AtomStorage.
    // Нужно для рендера, UI, сохранения — не для каждого шага.
    void downloadState(AtomStorage& atoms);

    // Геттеры для отладки / верификации
    GpuAtomBuffers& atomBuffers() { return atomBufs_; }
    GpuGridBuffers& gridBuffers() { return gridBufs_; }

private:
    void clearForcesAndEnergy(wgpu::CommandEncoder enc, uint32_t atomCount);
    void rebuildNeighborList(const AtomStorage& atoms, NeighborList& neighborList, const World& box, uint32_t atomCount);

    // ── GPU-подсистемы ───────────────────────────────────────────────────────
    GpuAtomBuffers atomBufs_;
    GpuGridBuffers gridBufs_;

    GpuVerletPredict gpuPredict_;
    GpuVerletCorrect gpuCorrect_;
    GpuStepOps gpuStepOps_;
    GpuSpatialGrid gpuSpatialGrid_;
    GpuWallForceField gpuWall_;
    GpuLJForceField gpuLJ_;

    // ── Кешированные параметры ───────────────────────────────────────────────
    Vec3f wallMax_ = {}; // box.size - Vec3f(1,1,1)
    Vec3f gravity_ = {};

    bool ready_ = false;
};
