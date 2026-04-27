#pragma once

#include "Benchmarks/BenchmarkCase.h"

class Simulation;
class World;

namespace Benchmarks {
    class BenchmarkScenes {
    public:
        static void build(Simulation& simulation, const BenchmarkCase& benchmarkCase);

    private:
        static void buildCrystal2D(World& world, const BenchmarkCase& benchmarkCase);
        static void buildCrystal3D(World& world, const BenchmarkCase& benchmarkCase);
        static void buildRandomGas2D(World& world, const BenchmarkCase& benchmarkCase);
    };
}
