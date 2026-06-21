from __future__ import annotations


def initialize_earth_engine(project: str | None = None):
    """Initialize Google Earth Engine lazily.

    Keeping this tiny wrapper avoids package-level Earth Engine side effects.
    Examples may call ``initialize_earth_engine(project='...')`` explicitly.
    """

    import ee

    if project is None:
        ee.Initialize()
    else:
        ee.Initialize(project=project)
        ee.data.setCloudApiUserProject(project)
    return ee


def ee_image_to_xarray_stack(image, *, geometry, crs: str, scale: float):
    """Convert an Earth Engine image to a ``band, y, x`` xarray stack.

    This is a thin wrapper around geemap and should remain optional through the
    ``hsa[earthengine]`` dependency group.
    """

    import geemap

    ds = geemap.ee_to_xarray(image, crs=crs, scale=scale, geometry=geometry)
    ds0 = ds.isel(time=0, drop=True).rename({"X": "x", "Y": "y"})
    return ds0.to_array(dim="band").transpose("band", "y", "x")
