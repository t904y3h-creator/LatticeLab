#pragma once

class GpuAtomBuffers;
class GpuVerletPredict;
class AtomStorage;
struct StepData;

class VerletScheme {
public:
    VerletScheme() = default;

    void pipeline(StepData& stepData) const;

    void setGpuPredict(GpuAtomBuffers* gpuBufs, GpuVerletPredict* gpuPredict);

    void correct(AtomStorage& atomStorage, float accelDamping, float dt) const;

private:
    void runGpuPredict(AtomStorage& atoms, float dt) const;

    GpuAtomBuffers* gpuBufs_ = nullptr;
    GpuVerletPredict* gpuPredict_ = nullptr;
};
