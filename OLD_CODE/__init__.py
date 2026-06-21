from .types import FeatureSpec
from .design_matrix import _build_design_matrix
from .sampling import _get_availability_domain, _get_sampling_points, _choose_chunk_size_points, _sample_env_layer
from .io import load_presence_csvs, to_reloc_gdf, reproject_reloc, to_reloc_gdf_projected
from .ee import init_ee, init_ee_on_client, make_aoi, aoi_to_ee, build_predictors_image, ee_image_to_env_xarray, save_env_zarr, load_env_zarr
from .dask_utils import shutdown_default_client, make_local_dask_client


from .rsf.model import fit_rsf, predict_rsf_points
from .rsf.surface import get_rsf_surface
from .rsf.eval import fixed_width_Boyce, sliding_window_Boyce
from .rsf.cv import cv_model
from .rsf.model_selection import eval_all_linear_candidates

__all__ = [
    "FeatureSpec",
    "_build_design_matrix",
    "_get_availability_domain",
    "_get_sampling_points",
    "_choose_chunk_size_points",
    "_sample_env_layer",
    "load_presence_csvs",
    "to_reloc_gdf",
    "reproject_reloc",
    "to_reloc_gdf_projected",
    "init_ee",
    "init_ee_on_client",
    "make_aoi",
    "aoi_to_ee",
    "build_predictors_image",
    "ee_image_to_env_xarray",
    "save_env_zarr",
    "load_env_zarr",
    "shutdown_default_client",
    "make_local_dask_client",
    
    "fit_rsf",
    "predict_rsf_points",
    "get_rsf_surface",
    "fixed_width_Boyce",
    "sliding_window_Boyce",
    "cv_model",
    "eval_all_linear_candidates"
]