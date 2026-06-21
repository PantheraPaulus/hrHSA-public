from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass(frozen=True)
class FeatureSpec:
    linear: List[str]
    quadratic: Optional[List[str]] = None
    interactions: Optional[List[Tuple[str, str]]] = None
    add_const: bool = True


@dataclass
class _FoldSplit:
    train: "gpd.GeoDataFrame"
    test: "gpd.GeoDataFrame"
    train_thin: "gpd.GeoDataFrame"
    test_thin: "gpd.GeoDataFrame"