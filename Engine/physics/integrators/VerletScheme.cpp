#include "VerletScheme.h"

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/GpuVerletPredict.h"
#include "Engine/metrics/Profiler.h"
#include "Engine/physics/integrators/StepOps.h"

void VerletScheme::setGpuPredict(GpuAtomBuffers* gpuBufs, GpuVerletPredict* gpuPredict) {
    gpuBufs_ = gpuBufs;
    gpuPredict_ = (gpuPredict && gpuPredict->isReady()) ? gpuPredict : nullptr;
}

void VerletScheme::pipeline(StepData& stepData) const {
    PROFILE_SCOPE("VerletScheme::pipeline");

    StepOps::predictAndSync(stepData, [this](AtomStorage& storage, float dt) { this->runGpuPredict(storage, dt); });
    StepOps::computeForces(stepData);
    correct(stepData.atomStorage, stepData.accelDamping, stepData.dt);
}

void VerletScheme::runGpuPredict(AtomStorage& atoms, float dt) const {
    PROFILE_SCOPE("VerletScheme::predict [GPU]");

    const uint32_t n = static_cast<uint32_t>(atoms.mobileCount());

    if (gpuBufs_->capacity() < atoms.capacity()) {
        gpuBufs_->resize(atoms.capacity());
    }

    gpuBufs_->uploadPositions(atoms, n);
    gpuBufs_->uploadVelocities(atoms, n);
    gpuBufs_->uploadForces(atoms, n);
    gpuBufs_->uploadInvMass(atoms, n);

    gpuPredict_->dispatch(*gpuBufs_, n, dt);

    gpuBufs_->downloadPositions(atoms, n);
}

void VerletScheme::correct(AtomStorage& atomStorage, float accelDamping, float dt) const {
    PROFILE_SCOPE("VerletScheme::correct");
    const size_t n = atomStorage.mobileCount();

    const float* RESTRICT fx = atomStorage.fxData();
    const float* RESTRICT fy = atomStorage.fyData();
    const float* RESTRICT fz = atomStorage.fzData();
    const float* RESTRICT pfx = atomStorage.pfxData();
    const float* RESTRICT pfy = atomStorage.pfyData();
    const float* RESTRICT pfz = atomStorage.pfzData();

    float* RESTRICT vx = atomStorage.vxData();
    float* RESTRICT vy = atomStorage.vyData();
    float* RESTRICT vz = atomStorage.vzData();

    const float* RESTRICT invMass = atomStorage.invMassData();

#pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        const float halfDtInvMass = 0.5f * accelDamping * dt * invMass[i];
        vx[i] += (pfx[i] + fx[i]) * halfDtInvMass;
        vy[i] += (pfy[i] + fy[i]) * halfDtInvMass;
        vz[i] += (pfz[i] + fz[i]) * halfDtInvMass;
    }
}