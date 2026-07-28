import rasterio as rio
from rasterio.features import rasterize

import rasterio as rio
from rasterio.features import rasterize

def observed_density_on_rsf_grid(
    points,
    template,
    *,
    value: float = 1.0,
):
    """Bin point counts onto the x/y grid of an RSF raster, no rasterio needed."""

    import numpy as np
    import xarray as xr

    # CRS handling if rioxarray metadata exists
    try:
        template_crs = template.rio.crs
    except Exception:
        template_crs = None

    if getattr(points, "crs", None) is not None and template_crs is not None:
        if points.crs != template_crs:
            points = points.to_crs(template_crs)

    xs = np.asarray(template["x"].values, dtype=float)
    ys = np.asarray(template["y"].values, dtype=float)

    xres = float(np.nanmedian(np.abs(np.diff(xs))))
    yres = float(np.nanmedian(np.abs(np.diff(ys))))

    x_edges = np.concatenate([
        [xs[0] - xres / 2],
        (xs[:-1] + xs[1:]) / 2,
        [xs[-1] + xres / 2],
    ])

    y_edges = np.concatenate([
        [ys[0] - yres / 2],
        (ys[:-1] + ys[1:]) / 2,
        [ys[-1] + yres / 2],
    ])

    # histogram2d needs increasing bin edges
    x_edges_sorted = np.sort(x_edges)
    y_edges_sorted = np.sort(y_edges)

    counts, _, _ = np.histogram2d(
        points.geometry.y.to_numpy(dtype=float),
        points.geometry.x.to_numpy(dtype=float),
        bins=[y_edges_sorted, x_edges_sorted],
    )

    # Re-orient to template y/x order
    if ys[0] > ys[-1]:
        counts = counts[::-1, :]

    if xs[0] > xs[-1]:
        counts = counts[:, ::-1]

    counts = counts * value

    out = xr.DataArray(
        counts.astype("float32"),
        coords={"y": template["y"], "x": template["x"]},
        dims=("y", "x"),
        name="observed_density",
    )

    if template_crs is not None:
        out = out.rio.write_crs(template_crs)

    return out

import matplotlib.pyplot as plt


def plot_heldout_spatial_boyce_diagnostic(
    heldout_id,
    diagnostics: dict,
    *,
    smooth_radius_cells: int | None = 3,
    cmap: str = "RdYlBu",
    vmin: float = -1,
    vmax: float = 1,
    figsize: tuple[float, float] = (13, 5),
):
    """Plot spatial rank residuals and Boyce curve for one held-out individual."""

    d = diagnostics[heldout_id]

    rsf = d["rsf"]
    domain = d["domain"]
    points = d["test_points"]
    boyce_bins = d["boyce_bins"]

    residual = make_rank_residual_map(
        rsf,
        points,
        domain,
        smooth_radius_cells=smooth_radius_cells,
    )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    ax = axes[0]

    residual.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": "Observed rank - RSF rank"},
    )

    if domain.crs != residual.rio.crs:
        domain_plot = domain.to_crs(residual.rio.crs)
    else:
        domain_plot = domain

    domain_plot.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1,
    )

    xmin, ymin, xmax, ymax = domain_plot.total_bounds

    pad_x = (xmax - xmin) * 0.08
    pad_y = (ymax - ymin) * 0.08

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")

    pts = points
    if pts.crs != residual.rio.crs:
        pts = pts.to_crs(residual.rio.crs)

    pts.plot(
        ax=ax,
        color="black",
        markersize=4,
        alpha=0.4,
    )

    ax.set_title(f"{heldout_id}: spatial RSF rank residual")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax = axes[1]

    curve = boyce_bins.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["rsf_mid", "pe"]
    )

    ax.plot(
        curve["rsf_mid"],
        curve["pe"],
        marker="o",
        linewidth=1.5,
    )

    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xlabel("Log RSF score")
    ax.set_ylabel("Validation use / available area")
    ax.set_title("Boyce curve")

    if "boyce" in d:
        boyce_value = d["boyce"]
    elif "boyce" in boyce_bins.columns:
        boyce_value = boyce_bins["boyce"].iloc[0]
    else:
        boyce_value = np.nan

    if np.isfinite(boyce_value):
        ax.text(
            0.98,
            0.02,
            f"Boyce = {boyce_value:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

    fig.tight_layout()
    return fig, axes

def make_rank_residual_map(
    rsf: xr.DataArray,
    points: gpd.GeoDataFrame,
    domain: gpd.GeoDataFrame | None = None,
    *,
    smooth_radius_cells: int | None = None,
) -> xr.DataArray:
    """Compare ranked observed point density to ranked RSF prediction.

    Positive values:
        observed use rank > RSF rank  → model underestimated use

    Negative values:
        RSF rank > observed use rank  → model overestimated use
    """

    if "band" in rsf.dims:
        rsf2 = rsf.squeeze(drop=True)
    else:
        rsf2 = rsf

    obs = observed_density_on_rsf_grid(points, rsf2)

    if smooth_radius_cells is not None and smooth_radius_cells > 0:
        obs = obs.rolling(
            x=smooth_radius_cells,
            y=smooth_radius_cells,
            center=True,
            min_periods=1,
        ).mean()

    # Optional: mask to domain
    if domain is not None:
        if domain.crs != rsf2.rio.crs:
            domain = domain.to_crs(rsf2.rio.crs)

        domain_mask = rasterize(
            [(geom, 1) for geom in domain.geometry],
            out_shape=(rsf2.sizes["y"], rsf2.sizes["x"]),
            transform=rsf2.rio.transform(),
            fill=0,
            dtype="uint8",
        ).astype(bool)

        rsf_vals = rsf2.where(domain_mask)
        obs_vals = obs.where(domain_mask)
    else:
        rsf_vals = rsf2
        obs_vals = obs

    # Rank only finite cells
    rsf_flat = rsf_vals.values.ravel()
    obs_flat = obs_vals.values.ravel()

    valid = np.isfinite(rsf_flat) & np.isfinite(obs_flat)

    rsf_rank = np.full_like(rsf_flat, np.nan, dtype="float32")
    obs_rank = np.full_like(obs_flat, np.nan, dtype="float32")

    rsf_rank[valid] = (
        pd.Series(rsf_flat[valid])
        .rank(pct=True)
        .to_numpy(dtype="float32")
    )

    obs_rank[valid] = (
        pd.Series(obs_flat[valid])
        .rank(pct=True)
        .to_numpy(dtype="float32")
    )

    residual = obs_rank - rsf_rank

    residual = residual.reshape(rsf2.sizes["y"], rsf2.sizes["x"])

    out = xr.DataArray(
        residual,
        coords={"y": rsf2["y"], "x": rsf2["x"]},
        dims=("y", "x"),
        name="rank_residual",
    ).rio.write_crs(rsf2.rio.crs)

    return out