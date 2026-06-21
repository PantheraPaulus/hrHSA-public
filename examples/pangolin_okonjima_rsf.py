"""Okonjima pangolin RSF example.

This file is the intended home for the project-specific workflow currently in
``RSF_pangolin.ipynb`` / ``RSF_pangolin.py``.

Keep the package code general, and keep project-specific choices here:
- file paths,
- reserve boundary,
- excluded buffers around houses/airfields/roads,
- Earth Engine project name,
- pangolin-specific predictor stack,
- final map/site-selection outputs.
"""

from hsa.types import FeatureSpec


def main():
    # 1. load telemetry and reserve perimeter
    # 2. clean/project relocations
    # 3. build or load environmental xarray stack
    # 4. sample used/available points
    # 5. fit and validate candidate RSFs
    # 6. predict RSF surface and export outputs
    spec = FeatureSpec(linear=["ndvi_mean_30m"], add_const=True)
    print("Pangolin RSF example scaffold. Starting spec:", spec)


if __name__ == "__main__":
    main()
