#include "VerletScheme.h"

#include "Engine/gpu/GpuAtomBuffers.h"
#include "Engine/gpu/GpuVerletCorrect.h"
#include "Engine/gpu/GpuVerletPredict.h"
#include "Engine/metrics/Profiler.h"
#include "Engine/physics/integrators/StepOps.h"

void VerletScheme::initGpu(GpuAtomBuffers* gpuBufs, GpuVerletPredict* gpuPredict, GpuVerletCorrect* gpuCorrect) {
    gpuBufs_ = gpuBufs;
    gpuPredict_ = (gpuPredict && gpuPredict->isReady()) ? gpuPredict : nullptr;
    gpuCorrect_ = (gpuCorrect && gpuCorrect->isReady()) ? gpuCorrect : nullptr;
}

void VerletScheme::pipeline(StepData& stepData) const {
    PROFILE_SCOPE("VerletScheme::pipeline");

    StepOps::predictAndSync(stepData, [this](AtomStorage& storage, float dt) { this->runGpuPredict(storage, dt); });
    StepOps::computeForces(stepData);

    runGpuCorrect(stepData);
}

void VerletScheme::runGpuPredict(AtomStorage& atoms, float dt) const {
    PROFILE_SCOPE("VerletScheme::predict [GPU]");

    const uint32_t n = static_cast<uint32_t>(atoms.mobileCount());

    if (gpuBufs_->countAtoms() < atoms.capacity()) {
        gpuBufs_->resize(atoms.capacity());
    }

    gpuBufs_->uploadPositions(atoms, n);
    gpuBufs_->uploadVelocities(atoms, n);
    gpuBufs_->uploadForces(atoms, n);
    gpuBufs_->uploadInvMass(atoms, n);

    // gpuPredict_->dispatch(*gpuBufs_, n, dt);

    gpuBufs_->downloadPositions(atoms, n);
}

void VerletScheme::runGpuCorrect(StepData& stepData) const {
    PROFILE_SCOPE("VerletScheme::correct [GPU]");

    AtomStorage& atoms = stepData.atomStorage;
    const uint32_t n = static_cast<uint32_t>(atoms.mobileCount());

    // Загружаем свежие CPU-силы и скорости.
    // invMass уже на GPU с этапа predict — не перезагружаем.
    gpuBufs_->uploadVelocities(atoms, n);
    gpuBufs_->uploadForces(atoms, n);
    gpuBufs_->uploadPrevForces(atoms, n);

    // gpuCorrect_->dispatch(*gpuBufs_, n, stepData.dt, stepData.accelDamping);

    gpuBufs_->downloadVelocities(atoms, n);
}
