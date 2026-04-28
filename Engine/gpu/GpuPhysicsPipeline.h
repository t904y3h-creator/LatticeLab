#pragma once

#include <webgpu/webgpu-raii.hpp>

#include "Engine/gpu/GpuStepOps.h"
#include "Engine/gpu/GpuVerletCorrect.h"
#include "Engine/gpu/GpuVerletPredict.h"
#include "Engine/gpu/force/GpuLJForceField.h"
#include "Engine/gpu/force/GpuWallForceField.h"
#include "Engine/gpu/neigbors/GpuSpatialGrid.h"

class World;

struct GpuStepParams {
    float dt;
    float accelDamping;
    float maxParticleSpeed; // 0 = отключено
    bool enableLJ;
};

// GpuPhysicsPipeline
//
// Владеет всеми GPU-ресурсами и оркестрирует полный шаг симуляции на GPU.
//
// Порядок шага:
//   1. predict        - обновить позиции
//   2. confineToBox   - отразить от стенок
//   3. clearForces    - обнулить силы/энергию
//   4. [Grid rebuild] - перестройка сетки
//   5. wallForces     - мягкие стены + гравитация
//   6. ljForces       - LJ попарные взаимодействия
//   7. correct        - обновить скорости
//   8. [velCap]       - ограничить максимальную скорость

class GpuPhysicsPipeline {
public:
    GpuPhysicsPipeline(const World& world);

    GpuPhysicsPipeline(const GpuPhysicsPipeline&) = delete;
    GpuPhysicsPipeline& operator=(const GpuPhysicsPipeline&) = delete;

    bool isReady() const { return ready_; }

    void step(wgpu::CommandEncoder encoder, const World& world, const GpuStepParams& params);

private:
    void clearForcesAndEnergy(wgpu::CommandEncoder enc, const World& world);
    void prepare(const World& world);
    wgpu::Buffer createSharedUniformBuffer();

    wgpu::Buffer sharedUniforms_;
    GpuVerletPredict gpuPredict_;
    GpuVerletCorrect gpuCorrect_;
    GpuStepOps gpuStepOps_;
    GpuSpatialGrid gpuSpatialGrid_;
    GpuWallForceField gpuWall_;
    GpuLJForceField gpuLJ_;

    bool ready_ = false;
};
