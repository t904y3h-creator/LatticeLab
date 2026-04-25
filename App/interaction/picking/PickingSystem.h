#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

#include "App/interaction/selection/OverlayState.h"
#include "Engine/math/Vec2.h"
#include "Engine/math/Vec3.h"

class IRenderer;
class World;

struct AtomHit {
    size_t index;
    float distance;
};

class PickingSystem {
public:
    PickingSystem(World& world, std::unique_ptr<IRenderer>& renderer);

    void clearSelection();

    bool pickAtom(Vec2i screenPos, float tolerance, AtomHit& hit) const;

    void processClick(Vec2i screenPos, bool cumulative = false);
    void processRect(Vec2i start, Vec2i end, bool cumulative = false);
    void processLasso(std::span<Vec2i> points, bool cumulative = false);

    void handleAtomRemoval(size_t removedIndex);

    const std::unordered_set<size_t>& getSelectedIndices() const { return selectedIndices; }
    const OverlayState& getOverlay() const { return overlay; }
    OverlayState& getOverlay() { return overlay; }

private:
    // Скачивает позиции и типы с GPU — вызывать только при пикинге
    void downloadAtomData() const;

    bool pickAtom2D(Vec2i screenPos, float tolerance, AtomHit& hit) const;
    bool pickAtom3D(Vec2i screenPos, AtomHit& hit) const;

    template <typename T> static bool pointInPolygon(Vec2<T> point, std::span<Vec2<T>> polygon);
    template <typename T> static bool pointInRect(Vec2<T> point, Vec2<T> start, Vec2<T> end);

    World& world;
    std::unique_ptr<IRenderer>* renderer;
    OverlayState overlay;
    std::unordered_set<size_t> selectedIndices;

    // CPU копия только для пикинга
    mutable std::vector<Vec3f> cachedPositions_;
    mutable std::vector<uint32_t> cachedTypes_;
};
