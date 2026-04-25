#pragma once

#include <string>

#include "Engine/World.h"
#include "Engine/gpu/GpuPhysicsPipeline.h"
#include "Engine/metrics/EnergyMetrics.h"

class Simulation {
public:
    explicit Simulation(World& world);

    void step();

    void setDt(float dt) { dt_ = dt; }
    float getDt() const { return dt_; }
    void setMaxParticleSpeed(float v) { maxParticleSpeed_ = v; }
    float getMaxParticleSpeed() const { return maxParticleSpeed_; }
    void setAccelDamping(float v) { accelDamping_ = v; }
    float getAccelDamping() const { return accelDamping_; }

    size_t getSimStep() const { return simStep_; }
    float simTimeNs() const { return simTimeNs_; }
    void restoreRuntimeState(size_t simStep, float simTimeNs) {
        simStep_ = simStep;
        simTimeNs_ = simTimeNs;
    }

    void setSceneTitle(std::string_view title) { sceneTitle_ = title; }
    const std::string& sceneTitle() const { return sceneTitle_; }
    void setSceneDescription(std::string_view description) { sceneDescription_ = description; }
    const std::string& sceneDescription() const { return sceneDescription_; }

    float averageKineticEnergyEv() const {
        refreshMetrics();
        return metricsCache_.averageKineticEnergyEv;
    }
    float averagePotentialEnergyEv() const {
        refreshMetrics();
        return metricsCache_.averagePotentialEnergyEv;
    }
    float fullAverageEnergyEv() const {
        refreshMetrics();
        return metricsCache_.fullAverageEnergyEv();
    }
    float temperatureK() const {
        refreshMetrics();
        return metricsCache_.temperatureK();
    }
    float temperatureC() const {
        refreshMetrics();
        return metricsCache_.temperatureC();
    }
    float averageSpeedKmPerHour() const {
        refreshMetrics();
        return metricsCache_.averageSpeedKmPerHour();
    }
    float fullEnergyPJ() const { return fullAverageEnergyEv() * static_cast<float>(world_.atomCount()) * Units::kEvToPJ; }

    void invalidateMetrics() { metricsCacheValid_ = false; }

    World& world() { return world_; }
    const World& world() const { return world_; }

private:
    void refreshMetrics() const;

    World& world_;
    GpuPhysicsPipeline pipeline_;

    float dt_ = 0.01f;
    float maxParticleSpeed_ = 0.f;
    float accelDamping_ = 0.9f;

    size_t simStep_ = 0;
    float simTimeNs_ = 0.f;

    std::string sceneTitle_;
    std::string sceneDescription_;

    mutable bool metricsCacheValid_ = false;
    mutable EnergyMetrics::Snapshot metricsCache_{};
};
