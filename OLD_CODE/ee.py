from __future__ import annotations
import geopandas as gpd
import xarray as xr
import numpy as np

def init_ee(project_id=None):
    import ee
    if project_id is None:
        ee.Initialize()
    else:
        ee.Initialize(project=project_id)
        ee.data.setCloudApiUserProject(project_id)
    return ee.Number(1).getInfo()
        
def init_ee_on_client(client, project_id=None):
    client.register_worker_callbacks(setup=lambda: init_ee(project_id))
    return client.run(init_ee, project_id)

def make_aoi(reloc: gpd.GeoDataFrame, buffer_m: float = 1.0) -> gpd.GeoDataFrame:
    geom = reloc.unary_union.envelope.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=[geom], crs=reloc.crs)

def aoi_to_ee(aoi: gpd.GeoDataFrame):
    import geemap
    return geemap.geopandas_to_ee(aoi)

def _mask_s2_sr(img):
    import ee
    qa = img.select("QA60")
    cloud = 1 << 10
    cirrus = 1 << 11
    mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))
    return img.updateMask(mask).divide(10000).copyProperties(img, ["system:time_start"])

def build_predictors_image(
    aoi_ee,
    start: str,
    end: str,
    cloud_pct: float = 30,
    scale_m: float = 30,
):
    import ee

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_ee)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_s2_sr)
    )

    ndvi = s2.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi")).median().clip(aoi_ee)
    ndwi = s2.map(lambda img: img.normalizedDifference(["B3", "B8"]).rename("ndwi")).median().clip(aoi_ee)

    altitude = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi_ee)
    slope = ee.Terrain.slope(altitude).rename("slope")

    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(aoi_ee)
    water = gsw.gte(10).selfMask()
    dist2water = (
        water.fastDistanceTransform(256).sqrt()
        .multiply(scale_m)
        .rename("dist2water")
        .clip(aoi_ee)
    )

    predictors_img = (
        ndvi.addBands([ndwi, slope, dist2water])
        .toFloat()
        .clip(aoi_ee)
    )
    return predictors_img

def ee_image_to_env_xarray(
    predictors_img,
    aoi_ee,
    crs: str = "EPSG:29333",
    scale: float = 100,
    chunk_xy: int = 1024,
) -> xr.DataArray:
    import geemap
    import rioxarray  

    ds = geemap.ee_to_xarray(predictors_img, crs=crs, scale=scale, geometry=aoi_ee.geometry())

    ds0 = (
        ds.isel(time=0, drop=True)
        .rename({"X": "x", "Y": "y"})
        .chunk({"y": chunk_xy, "x": chunk_xy})
    )

    env = (
        ds0.to_array(dim="band")
        .transpose("band", "y", "x")
        .chunk({"band": -1, "y": chunk_xy, "x": chunk_xy})
        .rio.write_crs(crs)
    )
    return env

def save_env_zarr(env: xr.DataArray, path: str, mode: str = "w") -> None:
    env.to_dataset(name="env").to_zarr(path, mode=mode)

def load_env_zarr(path: str, var: str = "env") -> xr.DataArray:
    return xr.open_zarr(path)[var]