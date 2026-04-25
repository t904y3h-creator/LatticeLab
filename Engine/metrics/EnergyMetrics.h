#pragma once

#include <vector>

#include "Engine/World.h"
#include "Engine/math/Vec3.h"
#include "Engine/physics/AtomData.h"

namespace EnergyMetrics {
    struct Snapshot {
        float averageKineticEnergyEv = 0.0f;
        float averagePotentialEnergyEv = 0.0f;
        float averageSpeedSim = 0.0f;

        float fullAverageEnergyEv() const { return averageKineticEnergyEv + averagePotentialEnergyEv; }
        float temperatureK() const { return 2.0f * averageKineticEnergyEv / (3.0f * Units::kboltzmann); }
        float temperatureC() const { return temperatureK() - 273.15f; }
        float averageSpeedKmPerHour() const { return averageSpeedSim * Units::SpeedUnitToKmph; }
    };

    inline float kineticEnergy(AtomData::Type type, const Vec3f& vel) { return 0.5f * AtomData::getProps(type).mass * vel.sqrAbs(); }

    // TODO сделать шейдер вычисляющий энергию
    inline Snapshot buildSnapshot(const World& world) {
        Snapshot snapshot;
        const size_t count = world.mobileCount();
        if (count == 0) {
            return snapshot;
        }

        std::vector<Vec3f> velocities(count);
        std::vector<float> pe(count);
        std::vector<uint32_t> types(count);

        // скачиваем только мобильные атомы (первые mobileCount)
        world.getAtomBuffers().downloadVelocities(velocities);
        world.getAtomBuffers().downloadPe(pe);
        world.getAtomBuffers().downloadAtomType(types);

        float totalKE = 0.f;
        float totalPE = 0.f;
        float totalSpeed = 0.f;

        for (size_t i = 0; i < count; ++i) {
            totalKE += kineticEnergy(static_cast<AtomData::Type>(types[i]), velocities[i]);
            totalPE += pe[i];
            totalSpeed += velocities[i].abs();
        }

        const float inv = 1.f / static_cast<float>(count);
        snapshot.averageKineticEnergyEv = totalKE * inv;
        snapshot.averagePotentialEnergyEv = totalPE * inv;
        snapshot.averageSpeedSim = totalSpeed * inv;
        return snapshot;
    }

    inline float averageKineticEnergy(const World& world) { return buildSnapshot(world).averageKineticEnergyEv; }
    inline float averagePotentialEnergy(const World& world) { return buildSnapshot(world).averagePotentialEnergyEv; }
    inline float fullAverageEnergy(const World& world) { return buildSnapshot(world).fullAverageEnergyEv(); }
}
