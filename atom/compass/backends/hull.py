# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The convex hull of the features a fit ranged over, as a law or residual
file holds it."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

#: A point on a facet evaluates to rounding error, relative to its magnitudes.
_HULL_TOLERANCE = 1e-9


class Hull:
    """The convex hull of the features a fit ranged over.

    `equal` maps each feature that was constant over the fit to its value;
    `facets` bound the others, in `features` order, as `a . x + c <= 0` inside.
    `label` names the hull in every load error.
    """

    def __init__(self, label: str, features: tuple[str, ...], hull: Mapping) -> None:
        equal = hull.get("equal", {})
        if not set(equal) <= set(features):
            raise ValueError(
                f"{label} checks {sorted(equal)} for equality, which "
                f"are not all among its features {list(features)}"
            )
        self.equal = {features.index(f): value for f, value in equal.items()}
        self.vary = [i for i in range(len(features)) if i not in self.equal]
        width = len(self.vary) + 1
        if any(len(row) != width for row in hull["facets"]):
            raise ValueError(
                f"{label} has a facet not {width} wide: one entry per "
                f"varying feature {[features[i] for i in self.vary]}, then the offset"
            )
        self.facets = np.asarray(hull["facets"], dtype=np.float64).reshape(-1, width)
        if self.vary and not len(self.facets):
            raise ValueError(
                f"{label} bounds {len(self.vary)} varying features with no "
                "facets, which is all of space"
            )
        if not np.isfinite(self.facets).all():
            raise ValueError(f"{label} has a facet that is not finite")

    def contains(self, point: tuple[float, ...]) -> bool:
        if any(point[i] != value for i, value in self.equal.items()):
            return False
        x = np.array([point[i] for i in self.vary], dtype=np.float64)
        normals, offsets = self.facets[:, :-1], self.facets[:, -1]
        scale = np.abs(normals) @ np.abs(x) + np.abs(offsets)
        return bool(np.all(normals @ x + offsets <= _HULL_TOLERANCE * scale))

    def excess(self, point: tuple[float, ...]) -> float:
        """How far `point` is outside: the most it differs from an equal
        feature or passes a facet, `a . x + c`; 0 inside. A fit's facets are
        hulled in the box of its features' ranges, so past two or more varying
        features it is a share of a range."""
        x = np.array([point[i] for i in self.vary], dtype=np.float64)
        passed = self.facets[:, :-1] @ x + self.facets[:, -1]
        differ = [abs(point[i] - value) for i, value in self.equal.items()]
        return float(max([0.0, *passed, *differ]))
