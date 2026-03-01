from __future__ import annotations

from glob import glob
from typing import Sequence

import pandas as pd
import geopandas as gpd


def load_presence_csvs(
    pattern: str = "data/*.csv",
    columns: Sequence[str] = ("Timestamp", "ID", "Latitude", "Longitude"),
) -> pd.DataFrame:
    rows = []
    for path in glob(pattern):
        rows.append(pd.read_csv(path)[list(columns)])
    if not rows:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(rows, ignore_index=True)


def to_reloc_gdf(
    reloc: pd.DataFrame,
    *,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    timestamp_col: str = "Timestamp",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        reloc,
        geometry=gpd.points_from_xy(reloc[lon_col], reloc[lat_col]),
        crs=crs,
    )
    gdf[timestamp_col] = pd.to_datetime(gdf[timestamp_col])
    return gdf


def reproject_reloc(
    reloc: gpd.GeoDataFrame,
    *,
    target_crs: str,
) -> gpd.GeoDataFrame:
    if reloc.crs is None:
        raise ValueError("reloc.crs is None; set it (or create via to_reloc_gdf) before reprojecting.")
    if str(reloc.crs) == str(target_crs):
        return reloc
    return reloc.to_crs(target_crs)


def to_reloc_gdf_projected(
    reloc: pd.DataFrame,
    *,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    timestamp_col: str = "Timestamp",
    input_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:29333",
) -> gpd.GeoDataFrame:
    gdf = to_reloc_gdf(
        reloc,
        lon_col=lon_col,
        lat_col=lat_col,
        timestamp_col=timestamp_col,
        crs=input_crs,
    )
    return reproject_reloc(gdf, target_crs=target_crs)


def _first_timestamp_by_id(
    reloc: gpd.GeoDataFrame,
    *,
    id_col: str = "ID",
    timestamp_col: str = "Timestamp",
) -> pd.Series:
    return reloc.groupby(id_col)[timestamp_col].min().sort_values()