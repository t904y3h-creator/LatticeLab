#include "Scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

namespace Scenes {
    namespace detail {
        bool isTooClose(const std::vector<Vec3f>& positions, const Vec3f& candidate, float minDistSqr) {
            for (const Vec3f& p : positions) {
                if ((p - candidate).sqrAbs() < minDistSqr) {
                    return true;
                }
            }
            return false;
        }

        uint32_t resolveSeed(uint32_t seed) { return seed == 0 ? std::random_device{}() : seed; }
    }

    void crystal(World& world, int n, AtomData::Type type, bool is3d, double padding, double margin) {
        const double side = n * padding + padding + 2.0 * margin;
        world.setWorldSize(Vec3f(side, side, is3d ? side : world.getWorldSize().z));

        const Vec3f vecMargin(margin, margin, is3d ? margin : 0.0);
        const int zMax = is3d ? n : 1;

        std::vector<Vec3f> positions;
        std::vector<Vec3f> velocities;
        std::vector<uint32_t> types;

        positions.reserve(static_cast<size_t>(n) * n * zMax);
        velocities.reserve(positions.capacity());
        types.reserve(positions.capacity());

        for (int x = 1; x <= n; ++x) {
            for (int y = 1; y <= n; ++y) {
                for (int z = 1; z <= zMax; ++z) {
                    positions.emplace_back(Vec3f(x, y, z) * padding + vecMargin);
                    velocities.emplace_back(Vec3f::Random() * 0.5f);
                    types.emplace_back(static_cast<uint32_t>(type));
                }
            }
        }

        world.getAtomBuffers().resize(positions.size());
        world.getAtomBuffers().uploadPositions(positions);
        world.getAtomBuffers().uploadVelocities(velocities);
        world.getAtomBuffers().uploadAtomType(types);
    }

    int randomGasInCurrentBox(World& world, int atomCount, AtomData::Type type, bool is3d, float minDistance, float speedScale,
                              int maxAttemptsPerAtom, uint32_t seed) {
        atomCount = std::max(0, atomCount);
        if (atomCount == 0) {
            return 0;
        }

        std::mt19937 rng(detail::resolveSeed(seed));
        std::uniform_real_distribution<float> distX(2.f, world.getWorldSize().x - 2.f);
        std::uniform_real_distribution<float> distY(2.f, world.getWorldSize().y - 2.f);
        std::uniform_real_distribution<float> distZ(2.f, world.getWorldSize().z - 2.f);

        const float minDistSqr = minDistance * minDistance;
        const float zMid = world.getWorldSize().z * 0.5f;

        std::vector<Vec3f> positions;
        std::vector<Vec3f> velocities;
        std::vector<uint32_t> types;
        positions.reserve(static_cast<size_t>(atomCount));

        for (int i = 0; i < atomCount; ++i) {
            for (int attempt = 0; attempt < maxAttemptsPerAtom; ++attempt) {
                const Vec3f candidate(distX(rng), distY(rng), is3d ? distZ(rng) : zMid);
                if (!detail::isTooClose(positions, candidate, minDistSqr)) {
                    positions.emplace_back(candidate);
                    const Vec3f spd = Vec3f::Random() * speedScale;
                    velocities.emplace_back(is3d ? spd : Vec3f(spd.x, spd.y, 0.f));
                    types.emplace_back(static_cast<uint32_t>(type));
                    break;
                }
            }
        }

        if (positions.empty()) {
            return 0;
        }

        world.getAtomBuffers().resize(positions.size());
        world.getAtomBuffers().uploadPositions(positions);
        world.getAtomBuffers().uploadVelocities(velocities);
        world.getAtomBuffers().uploadAtomType(types);

        return static_cast<int>(positions.size());
    }

    void randomGas(World& world, int atomCount, AtomData::Type type, bool is3d, double spacing, double margin, float density,
                   float speedScale, uint32_t seed) {
        atomCount = std::max(0, atomCount);
        const float clampedDensity = std::clamp(density, 0.25f, 3.0f);
        const double effectiveSpacing = spacing / static_cast<double>(clampedDensity);

        const int sideCount = is3d ? std::max(1, static_cast<int>(std::ceil(std::cbrt(static_cast<double>(atomCount)))))
                                   : std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(atomCount)))));

        const double span = sideCount * effectiveSpacing + 2.0 * margin;
        world.setWorldSize(Vec3f(span, span, is3d ? span : 6));

        randomGasInCurrentBox(world, atomCount, type, is3d, 4.f, speedScale, 20, seed);
    }
}
