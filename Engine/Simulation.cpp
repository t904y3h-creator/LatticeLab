#include "Simulation.h"

Simulation::Simulation(World& world) : world_(world), pipeline_(world) {}

void Simulation::step() {
    if (!pipeline_.isReady() || world_.atomCount() == 0) {
        return;
    }

    GpuStepParams params{};
    params.dt = dt_;
    params.accelDamping = accelDamping_;
    params.maxParticleSpeed = maxParticleSpeed_;

    pipeline_.step(world_, params);

    simStep_++;
    simTimeNs_ += dt_;
    invalidateMetrics();
}

void Simulation::refreshMetrics() const {
    if (metricsCacheValid_) {
        return;
    }
    metricsCache_ = EnergyMetrics::buildSnapshot(world_);
    metricsCacheValid_ = true;
}
