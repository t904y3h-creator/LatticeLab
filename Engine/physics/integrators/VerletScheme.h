#pragma once

class GpuAtomBuffers;
class GpuVerletPredict;
class GpuVerletCorrect;
class AtomStorage;
struct StepData;

class VerletScheme {
public:
    VerletScheme() = default;

    void pipeline(StepData& stepData) const;

    void initGpu(GpuAtomBuffers* gpuBufs, GpuVerletPredict* gpuPredict, GpuVerletCorrect* gpuCorrect);

private:
    void runGpuPredict(AtomStorage& atoms, float dt) const;
    void runGpuCorrect(StepData& stepData) const;

    GpuAtomBuffers* gpuBufs_ = nullptr;
    GpuVerletPredict* gpuPredict_ = nullptr;
    GpuVerletCorrect* gpuCorrect_ = nullptr;
};
