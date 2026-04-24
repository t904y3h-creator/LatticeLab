#pragma once

#include <cmath>

#include "Engine/SimBox.h"
#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/GpuStepOps.h"
#include "Engine/metrics/Profiler.h"
#include "Engine/physics/ForceField.h"
#include "Engine/physics/Integrator.h"
#include "Engine/restrict.h"

class NeighborList;

namespace StepOps {
    inline GpuStepOps* gpuStepOps_ = nullptr;
    inline GpuAtomBuffers* gpuBufs_ = nullptr;

    inline void init(GpuAtomBuffers* gpuBufs, GpuStepOps* gpuStepOps) {
        gpuBufs_ = gpuBufs;
        gpuStepOps_ = gpuStepOps;
    }

    template <typename T>
    concept AtomStepFunc = requires(T fn, AtomStorage& storage, float dt) {
        { fn(storage, dt) } -> std::same_as<void>;
    };

    inline void confineToBox(AtomStorage& atomStorage, World& box) {
        const Vec3f max = box.size - Vec3f(1, 1, 1);
        // gpuStepOps_->dispatchConfine(*gpuBufs_, atomStorage.size(), max.x, max.y, max.z);

        // constexpr float restitution = 0.8f;
        // const Vec3f max = box.size - Vec3f(1.0, 1.0, 1.0);

        // auto confineAxis = [&](float& coord, float& speed, float axisMax) {
        //     if (coord < 0.0f) {
        //         coord = 0.0f;
        //         if (speed < 0.0f) {
        //             speed = -speed * restitution;
        //         }
        //     }
        //     else if (coord > axisMax) {
        //         coord = axisMax;
        //         if (speed > 0.0f) {
        //             speed = -speed * restitution;
        //         }
        //     }
        // };

        // for (size_t atomIndex = 0; atomIndex < atomStorage.mobileCount(); ++atomIndex) {
        //     confineAxis(atomStorage.posX(atomIndex), atomStorage.velX(atomIndex), static_cast<float>(max.x));
        //     confineAxis(atomStorage.posY(atomIndex), atomStorage.velY(atomIndex), static_cast<float>(max.y));
        //     confineAxis(atomStorage.posZ(atomIndex), atomStorage.velZ(atomIndex), static_cast<float>(max.z));
        // }
    }

    inline void postProcessVelocities(AtomStorage& atomStorage, float maxSpeed) {
        // gpuStepOps_->dispatchVelCap(*gpuBufs_, atomStorage.size(), maxSpeed);

        //         const float maxSpeedSqr = maxSpeed * maxSpeed;
        //         float* RESTRICT vx = atomStorage.vxData();
        //         float* RESTRICT vy = atomStorage.vyData();
        //         float* RESTRICT vz = atomStorage.vzData();

        //         const size_t mobileCount = atomStorage.mobileCount();
        // #pragma GCC ivdep
        //         for (size_t atomIndex = 0; atomIndex < mobileCount; ++atomIndex) {
        //             float vxValue = vx[atomIndex];
        //             float vyValue = vy[atomIndex];
        //             float vzValue = vz[atomIndex];

        //             const float speedSqr = vxValue * vxValue + vyValue * vyValue + vzValue * vzValue;
        //             if (speedSqr > maxSpeedSqr) {
        //                 const float scale = maxSpeed / std::sqrt(speedSqr);
        //                 vxValue *= scale;
        //                 vyValue *= scale;
        //                 vzValue *= scale;
        //             }

        //             vx[atomIndex] = vxValue;
        //             vy[atomIndex] = vyValue;
        //             vz[atomIndex] = vzValue;
        //         }
    }

    inline void computeForces(StepData& stepData) {
        PROFILE_SCOPE("StepOps::computeForces");
        stepData.forceField.compute(stepData.atomStorage, stepData.bonds, stepData.box, stepData.neighborList, stepData.allowBondFormation,
                                    stepData.dt);
    }

    template <typename StepFn>
        requires AtomStepFunc<StepFn>
    inline void predictAndSync(StepData& stepData, StepFn predictFn) {
        predictFn(stepData.atomStorage, stepData.dt);

        confineToBox(stepData.atomStorage, stepData.box);

        stepData.atomStorage.swapPrevCurrentForces();
        std::fill_n(stepData.atomStorage.fxData(), stepData.atomStorage.size(), 0.0f);
        std::fill_n(stepData.atomStorage.fyData(), stepData.atomStorage.size(), 0.0f);
        std::fill_n(stepData.atomStorage.fzData(), stepData.atomStorage.size(), 0.0f);
        std::fill_n(stepData.atomStorage.energyData(), stepData.atomStorage.size(), 0.0f);
    }
}
