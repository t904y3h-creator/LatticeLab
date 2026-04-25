#pragma once

#include <array>
#include <cstddef>

#include "Engine/physics/AtomData.h"

class LJTable {
public:
    struct LJParams {
        float potentialC6 = 0.0f;
        float potentialC12 = 0.0f;
    };

    static constexpr size_t TypeCount = static_cast<size_t>(AtomData::Type::COUNT);
    using LJPairTable = std::array<std::array<LJParams, TypeCount>, TypeCount>;
    using LJPairRow = std::array<LJParams, TypeCount>;

    LJTable();

    [[nodiscard]] const LJPairRow& pairRow(AtomData::Type type) const;

private:
    static LJPairTable buildLJPairTable();

    LJPairTable ljPairTable_;
};
