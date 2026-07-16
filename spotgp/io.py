"""HDF5 save/load for GPSolver and MCMC sampler state."""

import logging
import warnings

import numpy as np

logger = logging.getLogger("spotgp")

__all__ = ["save_gp", "load_gp", "save_sampler", "load_sampler"]


def _require_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O. Install with: "
            "pip install h5py  (or: pip install spotgp[hdf5])")


def _is_hdf5(path):
    return str(path).endswith((".h5", ".hdf5"))


# ── Class registry ────────────────────────────────────────────────────────

_CLASS_REGISTRY = None


def _get_class_registry():
    global _CLASS_REGISTRY
    if _CLASS_REGISTRY is not None:
        return _CLASS_REGISTRY

    from .envelope import (
        TrapezoidSymmetricEnvelope, TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope, ExponentialEnvelope,
        ExponentialAsymmetricEnvelope,
    )
    from .visibility import (
        VisibilityFunction, EdgeOnVisibilityFunction,
        FullGeometryVisibilityFunction,
    )
    from .latitude import (
        LatitudeDistributionFunction, UniformDoubleHemisphereBand,
    )

    _CLASS_REGISTRY = {
        "TrapezoidSymmetricEnvelope": TrapezoidSymmetricEnvelope,
        "TrapezoidAsymmetricEnvelope": TrapezoidAsymmetricEnvelope,
        "SkewedGaussianEnvelope": SkewedGaussianEnvelope,
        "ExponentialEnvelope": ExponentialEnvelope,
        "ExponentialAsymmetricEnvelope": ExponentialAsymmetricEnvelope,
        "VisibilityFunction": VisibilityFunction,
        "EdgeOnVisibilityFunction": EdgeOnVisibilityFunction,
        "FullGeometryVisibilityFunction": FullGeometryVisibilityFunction,
        "LatitudeDistributionFunction": LatitudeDistributionFunction,
        "UniformDoubleHemisphereBand": UniformDoubleHemisphereBand,
    }
    return _CLASS_REGISTRY


# ── Helpers ────────────────────────────────────────────────────────────────

def _replace_group(f, name):
    if name in f:
        del f[name]
    return f.create_group(name)


def _write_string_dataset(grp, name, strings):
    import h5py
    dt = h5py.string_dtype()
    grp.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


# ── Writers ────────────────────────────────────────────────────────────────

def _write_data(f, data):
    grp = _replace_group(f, "data")
    grp.create_dataset("x", data=np.asarray(data.x))
    grp.create_dataset("y", data=np.asarray(data.y))
    grp.create_dataset("yerr", data=np.asarray(data.yerr))


def _write_model(f, model):
    grp = _replace_group(f, "model")
    grp.attrs["sigma_k"] = model.sigma_k
    grp.attrs["fspot"] = model.fspot
    grp.attrs["alpha_max"] = model.alpha_max if model.alpha_max is not None else np.nan

    # Envelope
    env = grp.create_group("envelope")
    if model.envelope is not None:
        env.attrs["class_name"] = type(model.envelope).__name__
        for k, v in model.envelope.param_dict.items():
            env.attrs[k] = float(v)
    else:
        env.attrs["class_name"] = "none"

    # Visibility
    vis = grp.create_group("visibility")
    if model.visibility is not None:
        vis.attrs["class_name"] = type(model.visibility).__name__
        for k, v in model.visibility.param_dict.items():
            vis.attrs[k] = float(v)
        from .visibility import FullGeometryVisibilityFunction
        if isinstance(model.visibility, FullGeometryVisibilityFunction):
            vis.attrs["alpha_ref"] = model.visibility.alpha_ref
            vis.attrs["n_lon"] = model.visibility.n_lon
    else:
        vis.attrs["class_name"] = "none"

    # Latitude
    lat = grp.create_group("latitude")
    lat.attrs["class_name"] = type(model.latitude_distribution).__name__
    for k, v in model.latitude_distribution.param_dict.items():
        lat.attrs[k] = float(v)


def _write_config(f, gp):
    grp = _replace_group(f, "config")
    grp.attrs["kernel_type"] = gp.kernel_type
    grp.attrs["mean_val"] = float(gp.mean_val)
    grp.attrs["fit_sigma_n"] = bool(gp.fit_sigma_n)
    grp.attrs["matrix_solver"] = gp.matrix_solver
    grp.attrs["bandwidth"] = int(gp.bandwidth) if hasattr(gp, "bandwidth") else -1
    grp.attrs["n_harmonics"] = int(gp.n_harmonics)
    grp.attrs["n_lat"] = int(gp.n_lat)
    grp.attrs["quadrature"] = gp.kernel.quadrature

    _write_string_dataset(grp, "param_keys", list(gp.param_keys))
    grp.create_dataset("bounds", data=np.asarray(gp.bounds))

    if gp.lat_range is not None:
        grp.create_dataset("lat_range", data=np.array(gp.lat_range))


def _write_fit_map(f, gp):
    if gp.map_estimate is None:
        return
    grp = _replace_group(f, "fits/map")
    grp.create_dataset("theta", data=np.asarray(gp.map_estimate))
    result = getattr(gp, "_map_result", None)
    if result is not None:
        grp.attrs["success"] = bool(getattr(result, "success", False))
        grp.attrs["fun"] = float(getattr(result, "fun", np.nan))
        grp.attrs["nit"] = int(getattr(result, "nit", 0))
        grp.attrs["message"] = str(getattr(result, "message", ""))


def _write_fit_acf(f, gp):
    theta = getattr(gp, "acf_fit_theta", None)
    if theta is None:
        return
    grp = _replace_group(f, "fits/acf")
    grp.create_dataset("theta", data=np.asarray(theta))
    lag_centers = getattr(gp, "_acf_lag_centers", None)
    if lag_centers is not None:
        grp.create_dataset("lag_centers", data=np.asarray(lag_centers))
    acf_data = getattr(gp, "_acf_data", None)
    if acf_data is not None:
        grp.create_dataset("acf_data", data=np.asarray(acf_data))
    result = getattr(gp, "_acf_fit_result", None)
    if result is not None:
        grp.attrs["fun"] = float(getattr(result, "fun", np.nan))


def _write_mass_matrix(f, gp):
    imm = getattr(gp, "inverse_mass_matrix", None)
    if imm is None:
        return
    grp = _replace_group(f, "fits/mass_matrix")
    grp.create_dataset("inverse_mass_matrix", data=np.asarray(imm))
    hess = getattr(gp, "_hessian", None)
    if hess is not None:
        grp.create_dataset("hessian", data=np.asarray(hess))
    fisher = getattr(gp, "_fisher_matrix", None)
    if fisher is not None:
        grp.create_dataset("fisher_matrix", data=np.asarray(fisher))


# ── Readers ────────────────────────────────────────────────────────────────

def _read_data(f):
    from .observations import TimeSeriesData
    grp = f["data"]
    return TimeSeriesData(
        grp["x"][:], grp["y"][:], grp["yerr"][:], normalize=False)


def _read_model(f):
    from .spot_model import SpotEvolutionModel
    registry = _get_class_registry()
    grp = f["model"]

    # Envelope
    env_name = grp["envelope"].attrs["class_name"]
    if env_name == "none":
        envelope = None
    else:
        cls = registry[env_name]
        params = {k: float(v) for k, v in grp["envelope"].attrs.items()
                  if k != "class_name"}
        envelope = cls(**params)

    # Visibility
    vis_name = grp["visibility"].attrs["class_name"]
    if vis_name == "none":
        visibility = None
    else:
        cls = registry[vis_name]
        skip = {"class_name", "alpha_ref", "n_lon"}
        params = {k: float(v) for k, v in grp["visibility"].attrs.items()
                  if k not in skip}
        if vis_name == "FullGeometryVisibilityFunction":
            params["alpha_ref"] = float(grp["visibility"].attrs["alpha_ref"])
            params["n_lon"] = int(grp["visibility"].attrs["n_lon"])
        visibility = cls(**params)

    # Latitude
    lat_name = grp["latitude"].attrs["class_name"]
    cls = registry[lat_name]
    if lat_name == "UniformDoubleHemisphereBand":
        lat_min = float(grp["latitude"].attrs["lat_min"])
        lat_max = float(grp["latitude"].attrs["lat_max"])
        latitude = cls(
            min_lat_deg=float(np.rad2deg(lat_min)),
            max_lat_deg=float(np.rad2deg(lat_max)))
    elif lat_name == "LatitudeDistributionFunction":
        latitude = cls()
    else:
        params = {k: float(v) for k, v in grp["latitude"].attrs.items()
                  if k != "class_name"}
        latitude = cls(**params)

    sigma_k = float(grp.attrs["sigma_k"])
    fspot = float(grp.attrs["fspot"])
    alpha_max_val = float(grp.attrs["alpha_max"])
    alpha_max = None if np.isnan(alpha_max_val) else alpha_max_val

    return SpotEvolutionModel(
        envelope=envelope, visibility=visibility,
        sigma_k=sigma_k, fspot=fspot, alpha_max=alpha_max,
        latitude_distribution=latitude)


def _read_config(f):
    grp = f["config"]
    param_keys = list(grp["param_keys"].asstr()[:])
    bounds_arr = grp["bounds"][:]
    bounds_dict = {k: tuple(bounds_arr[i]) for i, k in enumerate(param_keys)}

    bw = int(grp.attrs["bandwidth"])
    lat_range = tuple(grp["lat_range"][:]) if "lat_range" in grp else None

    return dict(
        kernel_type=str(grp.attrs["kernel_type"]),
        mean=float(grp.attrs["mean_val"]),
        fit_sigma_n=bool(grp.attrs["fit_sigma_n"]),
        matrix_solver=str(grp.attrs["matrix_solver"]),
        bandwidth=bw if bw >= 0 else None,
        bounds=bounds_dict,
        n_harmonics=int(grp.attrs["n_harmonics"]),
        n_lat=int(grp.attrs["n_lat"]),
        quadrature=str(grp.attrs["quadrature"]),
        lat_range=lat_range,
    )


def _read_fit_results(f, gp):
    import jax.numpy as jnp
    from scipy.optimize import OptimizeResult

    if "fits/map" in f:
        grp = f["fits/map"]
        gp.map_estimate = jnp.asarray(grp["theta"][:])
        gp._map_result = OptimizeResult(
            x=np.asarray(grp["theta"][:]),
            success=bool(grp.attrs.get("success", False)),
            fun=float(grp.attrs.get("fun", np.nan)),
            nit=int(grp.attrs.get("nit", 0)),
            message=str(grp.attrs.get("message", "")),
        )

    if "fits/acf" in f:
        grp = f["fits/acf"]
        gp.acf_fit_theta = jnp.asarray(grp["theta"][:])
        if "lag_centers" in grp:
            gp._acf_lag_centers = grp["lag_centers"][:]
        if "acf_data" in grp:
            gp._acf_data = grp["acf_data"][:]
        gp._acf_fit_result = OptimizeResult(
            x=np.asarray(grp["theta"][:]),
            fun=float(grp.attrs.get("fun", np.nan)),
            success=True, nit=0, message="loaded from HDF5",
        )

    if "fits/mass_matrix" in f:
        grp = f["fits/mass_matrix"]
        gp.inverse_mass_matrix = jnp.asarray(grp["inverse_mass_matrix"][:])
        if "hessian" in grp:
            gp._hessian = jnp.asarray(grp["hessian"][:])
        if "fisher_matrix" in grp:
            gp._fisher_matrix = jnp.asarray(grp["fisher_matrix"][:])


# ── Sampler writers ────────────────────────────────────────────────────────

def _write_sampler(f, sampler):
    grp = _replace_group(f, "sampler")
    sampler_type = type(sampler).__name__
    grp.attrs["sampler_type"] = sampler_type

    # MAP solutions
    if (hasattr(sampler, "all_theta_maps")
            and sampler.all_theta_maps is not None
            and len(sampler.all_theta_maps) > 0):
        map_grp = grp.create_group("map_solutions")
        param_keys = list(sampler.param_keys)

        all_arrays = []
        for tm in sampler.all_theta_maps:
            if isinstance(tm, dict):
                arr = np.array([float(tm.get(k, np.nan)) for k in param_keys])
            else:
                arr = np.asarray(tm, dtype=np.float64)
            all_arrays.append(arr)
        map_grp.create_dataset("all_theta_maps",
                               data=np.array(all_arrays, dtype=np.float64))

        if hasattr(sampler, "map_loglikes") and sampler.map_loglikes is not None:
            map_grp.create_dataset("map_loglikes",
                                   data=np.asarray(sampler.map_loglikes))

    # NUTS state (BlackJAX)
    if sampler_type == "BlackJAXSampler" and sampler._last_state is not None:
        st = grp.create_group("nuts_state")
        st.create_dataset("position",
                          data=np.asarray(sampler._last_state.position))
        st.create_dataset("logdensity",
                          data=np.asarray(sampler._last_state.logdensity))
        st.create_dataset("logdensity_grad",
                          data=np.asarray(sampler._last_state.logdensity_grad))
        st.create_dataset("step_size",
                          data=np.asarray(sampler._adapted_step_size))
        st.create_dataset("inverse_mass_matrix",
                          data=np.asarray(sampler._adapted_inv_mass))
        st.create_dataset("rng_key",
                          data=np.asarray(sampler._last_rng_key))
        n_warmup = (sampler._info["n_warmup"]
                    if sampler._info is not None else 0)
        st.attrs["n_warmup"] = int(n_warmup)
        st.attrs["n_chains"] = int(getattr(sampler, "_n_chains", 1))

    # Dynesty results
    if sampler_type == "DynestySampler":
        dyn = grp.create_group("dynesty_results")
        info = getattr(sampler, "_info", None) or {}
        dyn.attrs["logz"] = float(info.get("logz", np.nan))
        dyn.attrs["logzerr"] = float(info.get("logzerr", np.nan))


def _append_samples_h5(f, samples, sampler_type="BlackJAXSampler"):
    if samples is None or (hasattr(samples, "size") and samples.size == 0):
        return 0

    samples = np.asarray(samples, dtype=np.float64)

    if "sampler" not in f:
        f.create_group("sampler")
    grp = f["sampler"]

    if "samples" not in grp:
        if samples.ndim == 2:
            maxshape = (None, samples.shape[1])
        else:
            maxshape = (samples.shape[0], None, samples.shape[2])
        grp.create_dataset("samples", data=samples,
                           maxshape=maxshape, chunks=True,
                           dtype=np.float64)
    else:
        ds = grp["samples"]
        if samples.ndim == 2:
            old_n = ds.shape[0]
            new_n = samples.shape[0]
            ds.resize(old_n + new_n, axis=0)
            ds[old_n:] = samples
        else:
            old_n = ds.shape[1]
            new_n = samples.shape[1]
            ds.resize(old_n + new_n, axis=1)
            ds[:, old_n:] = samples

    ds = grp["samples"]
    return ds.shape[0] if ds.ndim == 2 else ds.shape[1]


# ── Sampler readers ────────────────────────────────────────────────────────

def _read_sampler_state(f, sampler):
    if "sampler" not in f:
        return

    grp = f["sampler"]
    sampler_type = grp.attrs.get("sampler_type", "")

    # MAP solutions
    if "map_solutions" in grp:
        mg = grp["map_solutions"]
        if "all_theta_maps" in mg:
            all_arr = mg["all_theta_maps"][:]
            param_keys = list(sampler.param_keys)
            sampler.all_theta_maps = [
                {k: float(row[i]) for i, k in enumerate(param_keys)}
                for row in all_arr
            ]
            if len(sampler.all_theta_maps) > 0:
                sampler.theta_map = sampler.all_theta_maps[0]
        if "map_loglikes" in mg:
            sampler.map_loglikes = mg["map_loglikes"][:]

    # NUTS state
    if "nuts_state" in grp and sampler_type == "BlackJAXSampler":
        import blackjax
        import jax.numpy as jnp

        st = grp["nuts_state"]
        sampler._last_state = blackjax.mcmc.hmc.HMCState(
            position=jnp.asarray(st["position"][:]),
            logdensity=jnp.asarray(st["logdensity"][()]),
            logdensity_grad=jnp.asarray(st["logdensity_grad"][:]),
        )

        n_chains = int(st.attrs.get("n_chains", 1))
        sampler._n_chains = n_chains
        if n_chains > 1:
            sampler._adapted_step_size = np.asarray(st["step_size"][:])
        else:
            sampler._adapted_step_size = float(st["step_size"][()])
        sampler._adapted_inv_mass = jnp.asarray(
            st["inverse_mass_matrix"][:])
        sampler._last_rng_key = jnp.asarray(st["rng_key"][:])

        n_warmup = int(st.attrs.get("n_warmup", 0))

        # Count samples on disk without loading them
        n_on_disk = 0
        if "samples" in grp:
            ds = grp["samples"]
            if ds.size > 0:
                n_on_disk = ds.shape[1] if ds.ndim == 3 else ds.shape[0]

        sampler.samples = None
        sampler._info = {
            "step_size": sampler._adapted_step_size,
            "n_warmup": n_warmup,
            "n_samples": n_on_disk,
            "n_chains": n_chains,
            "n_divergent": 0,
        }
        sampler._warmup_completed = True

        logger.info("Checkpoint loaded from HDF5 (%d samples on disk, "
                     "%d chain(s))", n_on_disk, n_chains)

    # Dynesty results
    if "dynesty_results" in grp and sampler_type == "DynestySampler":
        dyn = grp["dynesty_results"]
        if sampler._info is None:
            sampler._info = {}
        sampler._info["logz"] = float(dyn.attrs.get("logz", np.nan))
        sampler._info["logzerr"] = float(dyn.attrs.get("logzerr", np.nan))

        if "samples" in grp:
            ds = grp["samples"]
            if ds.size > 0:
                sampler.samples = ds[:]

        logger.info("Dynesty results loaded from HDF5")


def _read_samples(f, flatten_chains=True):
    if "sampler" not in f or "samples" not in f["sampler"]:
        return None
    ds = f["sampler"]["samples"]
    if ds.size == 0:
        return None
    samples = ds[:]
    if flatten_chains and samples.ndim == 3:
        n_chains, n_samples, n_params = samples.shape
        samples = samples.reshape(n_chains * n_samples, n_params)
    return samples


# ── Public API ─────────────────────────────────────────────────────────────

def save_gp(path, gp):
    """Save GPSolver state to an HDF5 file.

    Writes data, model, config, and any completed fit results.
    Can be called incrementally — each call overwrites only the
    groups whose data has changed.

    Parameters
    ----------
    path : str
        File path (should end in ``.h5`` or ``.hdf5``).
    gp : GPSolver
        The solver to save.
    """
    h5py = _require_h5py()
    with h5py.File(path, "a") as f:
        _write_data(f, gp.data)
        _write_model(f, gp.spot_model)
        _write_config(f, gp)
        _write_fit_map(f, gp)
        _write_fit_acf(f, gp)
        _write_mass_matrix(f, gp)
    logger.info("GPSolver saved to %s", path)


def load_gp(path):
    """Reconstruct a GPSolver from an HDF5 file.

    The returned solver has all fit results (MAP, ACF, mass matrix)
    restored but is NOT JIT-compiled — call ``.build_jax()`` before
    running fits or sampling.

    Parameters
    ----------
    path : str
        Path to an HDF5 file written by :func:`save_gp`.

    Returns
    -------
    GPSolver
    """
    h5py = _require_h5py()
    from .gp_solver import GPSolver

    with h5py.File(path, "r") as f:
        data = _read_data(f)
        model = _read_model(f)
        config = _read_config(f)

        kernel_kwargs = {}
        for k in ("n_harmonics", "n_lat", "quadrature", "lat_range"):
            if k in config:
                val = config.pop(k)
                if val is not None:
                    kernel_kwargs[k] = val

        gp = GPSolver(data, model, **config, **kernel_kwargs)
        _read_fit_results(f, gp)

    logger.info("GPSolver loaded from %s", path)
    return gp


def save_sampler(path, sampler, append_samples=True):
    """Save MCMC sampler state to an HDF5 file.

    Parameters
    ----------
    path : str
        HDF5 file path (typically the same file used by ``save_gp``).
    sampler : MCMCSampler
        The sampler whose state to save.
    append_samples : bool
        If True, append in-memory samples to any existing samples
        in the file, then clear ``sampler.samples`` from memory.
    """
    h5py = _require_h5py()
    samples_to_save = (np.asarray(sampler.samples)
                       if sampler.samples is not None else None)

    with h5py.File(path, "a") as f:
        _write_sampler(f, sampler)
        if append_samples:
            n_on_disk = _append_samples_h5(
                f, samples_to_save, type(sampler).__name__)
        else:
            if "sampler" in f and "samples" in f["sampler"]:
                del f["sampler"]["samples"]
            n_on_disk = _append_samples_h5(
                f, samples_to_save, type(sampler).__name__)

    if append_samples and samples_to_save is not None:
        sampler.samples = None
        if sampler._info is not None:
            sampler._info = {
                "step_size": sampler._info.get("step_size"),
                "n_warmup": sampler._info.get("n_warmup", 0),
                "n_samples": n_on_disk,
                "n_divergent": sampler._info.get("n_divergent", 0),
            }

    logger.info("Sampler checkpoint saved to %s (%d samples on disk)",
                path, n_on_disk)


def load_sampler(path, sampler):
    """Restore MCMC sampler state from an HDF5 file.

    Parameters
    ----------
    path : str
        HDF5 file path.
    sampler : MCMCSampler
        The sampler to restore state into.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        _read_sampler_state(f, sampler)


def load_samples(path, flatten_chains=True):
    """Read MCMC samples from an HDF5 file.

    Parameters
    ----------
    path : str
        HDF5 file path.
    flatten_chains : bool
        If True and samples are 3-D (multi-chain), reshape to 2-D.

    Returns
    -------
    samples : ndarray or None
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        return _read_samples(f, flatten_chains=flatten_chains)
