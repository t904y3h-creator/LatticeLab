#include "BenchmarkScenes.h"

#include <algorithm>
#include <cmath>

#include "App/Scenes.h"
#include "Engine/Simulation.h"

namespace Benchmarks {
    int squareSideFromCount(int atomCount) { return std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(atomCount))))); }

    int cubeSideFromCount(int atomCount) { return std::max(1, static_cast<int>(std::ceil(std::cbrt(static_cast<double>(atomCount))))); }

    void BenchmarkScenes::build(Simulation& simulation, const BenchmarkCase& benchmarkCase) {
        simulation.world().clear();
        // simulation.setIntegrator(benchmarkCase.integrator);

        switch (benchmarkCase.scene) {
        case SceneKind::Crystal2D:
            buildCrystal2D(simulation.world(), benchmarkCase);
            break;
        case SceneKind::Crystal3D:
            buildCrystal3D(simulation.world(), benchmarkCase);
            break;
        case SceneKind::RandomGas2D:
            buildRandomGas2D(simulation.world(), benchmarkCase);
            break;
        }
    }

    void BenchmarkScenes::buildCrystal2D(World& world, const BenchmarkCase& benchmarkCase) {
        const int side = squareSideFromCount(benchmarkCase.atomCount);
        Scenes::crystal(world, side, AtomData::Type::Z, false);
    }

    void BenchmarkScenes::buildCrystal3D(World& world, const BenchmarkCase& benchmarkCase) {
        const int side = cubeSideFromCount(benchmarkCase.atomCount);
        Scenes::crystal(world, side, AtomData::Type::Z, true);
    }

    void BenchmarkScenes::buildRandomGas2D(World& world, const BenchmarkCase& benchmarkCase) {
        world.setWorldSize(benchmarkCase.boxSize);
        world.setGridCellSize(benchmarkCase.cellSize);
        Scenes::randomGasInCurrentBox(world, benchmarkCase.atomCount, AtomData::Type::H, false);
    }
}
