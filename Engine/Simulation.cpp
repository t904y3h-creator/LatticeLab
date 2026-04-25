#include "Simulation.h"

#include <cmath>

#include "Engine/io/SimulationStateIO.h"
#include "Engine/metrics/Profiler.h"
#include "Engine/physics/Bond.h"
#include "Engine/physics/Integrator.h"
#include "Engine/physics/integrators/StepOps.h"

Simulation::Simulation(World& box) : sim_box_(box), integrator() {
    atomStorage_.reserve(250000);
    // gpuBufs_.resize(250000);
    neighborList_.setParams(5.f, 1.f);
    forceField_.syncWalls(sim_box_);
}

void Simulation::enableGpu(bool enable) {
    if (enable) {
        gpuPipeline_.init(atomStorage_, forceField_, sim_box_);
    }
    // auto& schemeVar = integrator.getSchemeImpl();

    // if (auto* scheme = std::get_if<VerletScheme>(&schemeVar)) {
    //     StepOps::init(&gpuBufs_, &gpuStepOps_);
    //     if (enable && gpuVerletPredict_.isReady() && gpuVerletCorrect_.isReady()) {
    //         scheme->initGpu(&gpuBufs_, &gpuVerletPredict_, &gpuVerletCorrect_);
    //     }
    //     else {
    //         scheme->initGpu(nullptr, nullptr, nullptr);
    //     }
    // }
    // else {
    //     std::cerr << "GPU version is only available for Verlet integrator" << std::endl;
    // }
}

void Simulation::refreshMetricsCache() const {
    if (metricsCacheValid_) {
        return;
    }

    metricsCache_ = EnergyMetrics::buildSnapshot(atomStorage_);
    metricsCacheValid_ = true;
}

StepData Simulation::makeStepData() {
    return StepData{
        .atomStorage = atomStorage_,
        .bonds = bonds_,
        .box = sim_box_,
        .forceField = forceField_,
        .neighborList = neighborList_,
        .allowBondFormation = bondFormationEnabled_,
        .accelDamping = integrator.accelDamping(),
        .dt = Dt,
    };
}

void Simulation::update() {
    PROFILE_SCOPE("Simulation::update");
    // if (neighborList_.needsRebuild(atomStorage_)) {
    //     neighborList_.build(atomStorage_, sim_box_);
    //     neighborList_.recordRebuild(sim_step);
    // }

    StepData stepData = makeStepData();
    // integrator.step(stepData);

    GpuStepParams stepParams{
        .dt = stepData.dt,
        .accelDamping = stepData.accelDamping,
        .maxParticleSpeed = 0,
        .enableLJ = true,
    };

    gpuPipeline_.step(stepData.atomStorage, stepData.neighborList, stepData.box, stepParams);

    invalidateMetricsCache();
    ++sim_step;
    sim_time_ns += Dt * Units::kTimeUnitToNs;
}

void Simulation::setSizeBox(Vec3f newSize, int cellSize) {
    const bool resized = sim_box_.setSizeBox(newSize, cellSize);
    if (resized) {
        forceField_.syncWalls(sim_box_);
        sim_box_.grid.rebuild(atomStorage_.xDataSpan(), atomStorage_.yDataSpan(), atomStorage_.zDataSpan());
        neighborList_.clear();
    }
}

bool Simulation::createAtom(Vec3f start_coords, Vec3f start_speed, AtomData::Type type, bool fixed) {
    atomStorage_.addAtom(start_coords, start_speed, type, fixed);
    invalidateMetricsCache();
    sim_box_.grid.rebuild(atomStorage_.xDataSpan(), atomStorage_.yDataSpan(), atomStorage_.zDataSpan());
    return true;
}

bool Simulation::removeAtom(size_t atomIndex) {
    if (atomIndex >= atomStorage_.size()) {
        return false;
    }

    const size_t lastIndex = atomStorage_.size() - 1;

    for (auto it = bonds_.begin(); it != bonds_.end();) {
        if (it->aIndex == atomIndex || it->bIndex == atomIndex) {
            if (it->aIndex == atomIndex && it->bIndex != atomIndex && it->bIndex < atomStorage_.size()) {
                ++atomStorage_.valenceCount(it->bIndex);
            }
            if (it->bIndex == atomIndex && it->aIndex != atomIndex && it->aIndex < atomStorage_.size()) {
                ++atomStorage_.valenceCount(it->aIndex);
            }
            it = bonds_.erase(it);
            continue;
        }

        if (atomIndex != lastIndex) {
            if (it->aIndex == lastIndex) {
                it->aIndex = atomIndex;
            }
            if (it->bIndex == lastIndex) {
                it->bIndex = atomIndex;
            }
        }

        ++it;
    }

    atomStorage_.removeAtom(atomIndex);
    invalidateMetricsCache();
    sim_box_.grid.rebuild(atomStorage_.xDataSpan(), atomStorage_.yDataSpan(), atomStorage_.zDataSpan());
    return true;
}

void Simulation::addBond(size_t aIndex, size_t bIndex) {
    if (aIndex >= atomStorage_.size() || bIndex >= atomStorage_.size()) {
        return;
    }

    Bond::CreateBond(bonds_, aIndex, bIndex, atomStorage_);
}

void Simulation::clear() {
    atomStorage_.clear();
    invalidateMetricsCache();
    bonds_.clear();
    sceneTitle_.clear();
    sceneDescription_.clear();
    sim_box_.grid.rebuild(atomStorage_.xDataSpan(), atomStorage_.yDataSpan(), atomStorage_.zDataSpan());
    neighborList_.clear();
    sim_step = 0;
    sim_time_ns = 0.0f;
}
