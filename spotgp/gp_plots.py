"""GP plotting methods, mixed into GPSolver."""

import numpy as np


class GPPlotsMixin:
    """Plotting methods for GPSolver."""

    def plot_prediction(self, theta=None, n_points=2000, n_sigma=(1, 2),
                        ax=None, data_color="k", model_color="r",
                        show_legend=True, xlim=None, ylim=None, 
                        model_label="GP mean", data_label="Data"):
        """
        Plot the GP posterior mean and uncertainty bands over the data.

        If ``theta`` is provided the GP is temporarily updated to those
        hyperparameters before predicting, so the prediction reflects the
        given parameter values rather than whatever was last set internally.

        Parameters
        ----------
        theta : dict or array_like, shape (6,), optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``KERNEL_HPARAM_KEYS``, a sampling-space dict with ``log_``-
            prefixed keys (e.g. ``log_sigma_k``), or a length-6 array.
            If None, uses the current internal hyperparameters.
        n_points : int
            Number of prediction points spanning the data baseline.
        n_sigma : int or sequence of int
            Which sigma levels to shade.  E.g. ``(1, 2)`` draws both
            ±1σ and ±2σ bands (default).  Pass a single int for one band.
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        data_color, model_color : str
            Colors for data points and model curve/bands.
        show_legend : bool
            Whether to draw a legend.
        xlim : tuple, optional
            Limits for the x-axis. If None, defaults to the data range.
        ylim : tuple, optional
            Limits for the y-axis. If None, defaults to the data range.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if theta is not None:
            phys = {}
            for k, v in (theta.items() if isinstance(theta, dict)
                         else zip(self.spot_model.param_keys, theta)):
                if isinstance(k, str) and k.startswith("log_"):
                    phys[k[4:]] = 10.0 ** float(v)
                else:
                    phys[k] = float(v)
            self.update_hparam(phys)

        xpred = np.linspace(float(self.x[0]), float(self.x[-1]), n_points)
        mu, var = self.predict(xpred)
        sigma = np.sqrt(np.maximum(var, 0.0))

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        ax.errorbar(np.asarray(self.x), np.asarray(self.y),
                    yerr=np.asarray(self.yerr),
                    fmt=".", color=data_color, capsize=0, alpha=0.5,
                    label=data_label)
        ax.plot(xpred, mu, color=model_color, lw=1.5, label=model_label)

        alphas = {1: 0.35, 2: 0.18, 3: 0.10}
        for ns in (n_sigma if hasattr(n_sigma, "__iter__") else (n_sigma,)):
            ax.fill_between(xpred, mu - ns * sigma, mu + ns * sigma,
                            color=model_color,
                            alpha=alphas.get(ns, 0.15),
                            label=rf"$\pm{ns}\sigma$")
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(float(self.x[0]), float(self.x[-1]))
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Time [days]", fontsize=22)
        ax.set_ylabel("Flux", fontsize=22)
        if show_legend:
            ax.legend()

        return ax

    def plot_samples(self, theta=None, xpred=None, n_samples=5,
                     n_points=2000, source="prior", rng=None,
                     ax=None, show_data=True, data_color="k",
                     sample_alpha=0.7, sample_lw=1.0, cmap="tab10",
                     show_legend=True, xlim=None, ylim=None):
        """
        Plot sampled lightcurves from the GP prior or posterior.

        Parameters
        ----------
        theta : dict or array_like, optional
            Kernel parameters (see ``sample_lightcurves``).
        xpred : array_like, optional
            Prediction times.  If None, ``n_points`` evenly spaced.
        n_samples : int
            Number of samples to draw and plot (default 5).
        n_points : int
            Number of prediction points when ``xpred`` is None.
        source : {'prior', 'posterior'}
            Sample from the GP prior or posterior (default 'prior').
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.
        ax : matplotlib Axes, optional
            Axes to plot on.  If None, creates a new figure.
        show_data : bool
            Whether to overlay the observed data (default True).
        data_color : str
            Color for data points (default 'k').
        sample_alpha : float
            Opacity for sample curves (default 0.7).
        sample_lw : float
            Line width for sample curves (default 1.0).
        cmap : str
            Matplotlib colormap name for sample colors (default 'tab10').
        show_legend : bool
            Whether to draw a legend (default True).
        xlim, ylim : tuple, optional
            Axis limits.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        xpred, samples = self.sample_lightcurves(
            theta=theta, xpred=xpred, n_samples=n_samples,
            n_points=n_points, source=source, rng=rng,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        colormap = plt.get_cmap(cmap)
        for i, sample in enumerate(samples):
            label = f"Sample {i + 1}" if i < 10 else None
            ax.plot(xpred, sample, color=colormap(i % colormap.N),
                    alpha=sample_alpha, lw=sample_lw, label=label)

        if show_data:
            ax.errorbar(np.asarray(self.x), np.asarray(self.y),
                        yerr=np.asarray(self.yerr),
                        fmt=".", color=data_color, capsize=0, alpha=0.4,
                        label="Data", zorder=0)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(float(self.x[0]), float(self.x[-1]))
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Time [days]", fontsize=22)
        ax.set_ylabel("Flux", fontsize=22)
        ax.set_title(f"GP {source} samples", fontsize=16)
        if show_legend:
            ax.legend()

        return ax

    def plot_acf(self, theta=None, tlags=None, n_bins=50, ax=None,
                 normalize=False, data_color="k", model_color="r",
                 show_legend=True, xlim=None, ylim=None, 
                 model_label="Analytic ACF", data_label="Data ACF"):
        """
        Plot the empirical ACF and optionally the analytic kernel.

        Parameters
        ----------
        theta : dict or array_like, shape (6,), optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``KERNEL_HPARAM_KEYS``, a sampling-space dict with ``log_``-
            prefixed keys (e.g. ``log_sigma_k``), or a length-6 array.
            If provided, the analytic kernel is overplotted.
        tlags : array_like, optional
            Bin edges for compute_acf. If None, linearly spaced from 0 to
            half the baseline with n_bins+1 edges.
        n_bins : int
            Number of lag bins (used when tlags is None).
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        normalize : bool
            If True (default), normalize both curves by the data variance
            so ACF(0) ≈ 1.
        xlim : tuple, optional
            Limits for the x-axis. If None, defaults to the data range.
        ylim : tuple, optional
            Limits for the y-axis. If None, defaults to the data range.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        lag_centers, acf_data = self.compute_acf(tlags=tlags, n_bins=n_bins,
                                                    normalize=normalize)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(lag_centers, acf_data, color=data_color, label=data_label)

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)
            lag_fine = np.linspace(0.0, float(tlags[-1]), 300)
            K_model = np.asarray(_kernel_eval(
                theta_arr, jnp.asarray(lag_fine),
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
                r_gamma_func=self.spot_model.get_r_gamma_func(),
                lat_weight_func=self.spot_model.get_lat_weight_func(),
                cn_sq_func=self.spot_model.get_cn_sq_func(self.n_harmonics)))
            if normalize:
                y_full = getattr(self.data, '_y_full', self.data.y)
                var = np.var(y_full)
                if var > 0:
                    K_model = K_model / var
            ax.plot(lag_fine, K_model, color=model_color, label=model_label)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(min(tlags), max(tlags))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Time lag [days]", fontsize=22)
        ax.set_ylabel("ACF" if normalize else "Autocovariance", fontsize=22)
        if show_legend:
            ax.legend()

        return ax


    def plot_psd(self, theta=None, n_freq=500, dt_kernel=None, ax=None,
                 data_color="k", model_color="r", show_legend=True,
                 xlim=None, ylim=None, model_label="Analytic PSD", 
                 data_label="Data Lomb-Scargle"):
        """
        Plot the empirical PSD (Lomb-Scargle) and optionally the analytic
        kernel PSD (FFT of the autocovariance function).

        Both curves are normalized so their integral over positive frequencies
        equals the data variance, making them directly comparable.

        Parameters
        ----------
        theta : dict or array_like, shape (6,), optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``KERNEL_HPARAM_KEYS``, a sampling-space dict with ``log_``-
            prefixed keys (e.g. ``log_sigma_k``), or a length-6 array.
            If provided, the analytic kernel PSD is overplotted.
        n_freq : int
            Number of frequency points for the Lomb-Scargle periodogram.
        dt_kernel : float, optional
            Time step [days] for evaluating the analytic kernel on a uniform
            grid before FFT.  Defaults to one-fifth of the median data spacing.
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        data_color, model_color : str
            Colors for the data and model curves.
        show_legend : bool
            Whether to draw a legend.
        xlim : tuple, optional
            Limits for the x-axis. If None, defaults to the data range.
        ylim : tuple, optional
            Limits for the y-axis. If None, defaults to the data range.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        x = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        var = float(np.mean(resid ** 2))

        baseline = float(x[-1] - x[0])
        dt_med = float(np.median(np.diff(x)))
        freq_min = 1.0 / baseline
        freq_max = 1.0 / (2.0 * dt_med)

        # Data PSD via TimeSeriesData
        freqs, psd_data = self.data.compute_psd(
            normalization="psd", n_freq=n_freq,
            freq_min=freq_min, freq_max=freq_max)
        # Normalize so ∫PSD df = var(data)
        integral = np.trapezoid(psd_data, freqs)
        if integral > 0:
            psd_data = psd_data * var / integral

        if ax is None:
            fig, ax = plt.subplots()

        ax.semilogy(freqs, psd_data, color=data_color, lw=0.8, label=data_label)

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)

            if dt_kernel is None:
                dt_kernel = dt_med / 5.0
            tau_grid = np.arange(0.0, baseline, dt_kernel)
            K = np.asarray(_kernel_eval(
                theta_arr, jnp.asarray(tau_grid),
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
                r_gamma_func=self.spot_model.get_r_gamma_func(),
                lat_weight_func=self.spot_model.get_lat_weight_func(),
                cn_sq_func=self.spot_model.get_cn_sq_func(self.n_harmonics)))
            # Extend to two-sided symmetric sequence, then rfft → one-sided PSD
            K_twosided = np.concatenate([K[::-1], K[1:]])
            psd_model = np.abs(np.fft.rfft(K_twosided)) * dt_kernel
            freqs_model = np.fft.rfftfreq(len(K_twosided), d=dt_kernel)
            # Restrict to the data frequency range and skip DC
            mask = (freqs_model > 0) & (freqs_model <= freq_max)
            fm, pm = freqs_model[mask], psd_model[mask]
            # Normalize so ∫PSD df = var(data)
            pm = pm * var / np.trapezoid(pm, fm)
            ax.semilogy(fm, pm, color=model_color, lw=1.5, label=model_label)
            
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(freq_min, freq_max)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Frequency [1/day]", fontsize=22)
        ax.set_ylabel("PSD", fontsize=22)
        if show_legend:
            ax.legend()

        return ax


    def plot_covariance_matrix(self, theta=None, ax=None, cmap="RdBu_r",
                               show_colorbar=True, vmax=None, nbins=50,
                               show=False, filename="covariance_matrix.png"):
        """
        Plot the GP covariance matrix K (signal only, no noise).

        Entries outside the banded support are set to zero, matching the
        ``cholesky_banded`` approximation.  The matrix is binned to
        ``nbins x nbins`` before plotting.  The bandwidth boundary is drawn
        as dashed lines, and band width plus matrix sparsity are annotated.

        Parameters
        ----------
        theta : dict or array_like, optional
            Kernel hyperparameters.  Accepts a physical dict with keys from
            ``param_keys``, a sampling-space dict with ``log_``-prefixed keys,
            or a raw array.  If None, uses the current ``self.hparam`` values.
        ax : matplotlib Axes, optional
            Axes to plot on.  If None, a new figure is created.
        cmap : str, optional
            Colormap name.  Defaults to ``"RdBu_r"`` (diverging, centred at
            zero).
        show_colorbar : bool, optional
            Whether to add a colorbar.  Default True.
        vmax : float, optional
            Symmetric color scale limit ``[-vmax, vmax]``.  If None, uses the
            maximum absolute value of the banded matrix.
        nbins : int, optional
            Bin the N x N matrix down to ``nbins x nbins`` by averaging
            non-overlapping blocks before plotting.  Default 50.
        show : bool, optional
            If True, call ``plt.show()``.  Default False.
        filename : str, optional
            Filename used when saving to ``save_dir``.
            Default ``"covariance_matrix.png"``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import os
        import matplotlib.pyplot as plt

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)
        else:
            theta_arr = self._theta_dict_to_phys_array(self.hparam)

        N = self.N
        b = self.bandwidth
        dt = float(self.x[1] - self.x[0]) if N > 1 else 1.0
        band_days = b * dt

        # Build banded K: evaluate only the b+1 diagonals, zero elsewhere
        K = np.zeros((N, N))
        for d in range(b + 1):
            i_idx = np.arange(N - d)
            j_idx = i_idx + d
            lags = jnp.abs(self.x[i_idx] - self.x[j_idx])
            K_diag = np.asarray(_kernel_eval(
                theta_arr, lags,
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
                r_gamma_func=self.spot_model.get_r_gamma_func(),
                lat_weight_func=self.spot_model.get_lat_weight_func(),
                cn_sq_func=self.spot_model.get_cn_sq_func(self.n_harmonics)))
            K[i_idx, j_idx] = K_diag
            if d > 0:
                K[j_idx, i_idx] = K_diag

        # Sparsity: fraction of entries outside the band
        n_nonzero = int(N) * (2 * int(b) + 1) - int(b) * (int(b) + 1)
        n_nonzero = min(n_nonzero, N * N)
        sparsity = 100.0 * (1.0 - n_nonzero / (N * N))

        # Bin down to nbins x nbins by block-averaging
        n_plot = min(nbins, N)
        block = N // n_plot
        n_trim = block * n_plot
        K_bin = (K[:n_trim, :n_trim]
                 .reshape(n_plot, block, n_plot, block)
                 .mean(axis=(1, 3)))

        del K

        if vmax is None:
            vmax = float(np.max(np.abs(K_bin)))

        # Bandwidth in binned-matrix units
        b_bin = b / block

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(K_bin, origin="upper", cmap=cmap,
                       vmin=-vmax, vmax=vmax, aspect="auto")
        if show_colorbar:
            plt.colorbar(im, ax=ax, label="Covariance")

        # Dashed lines marking the bandwidth boundary in binned coordinates
        M = n_plot
        diag_x = np.array([-0.5, M - 0.5])
        ax.plot(diag_x + b_bin, diag_x, color="k", lw=1, ls="--", alpha=0.6)
        ax.plot(diag_x - b_bin, diag_x, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xlim(-0.5, M - 0.5)
        ax.set_ylim(M - 0.5, -0.5)

        # Sparsity annotation in upper-right corner
        ax.text(0.98, 0.02, f"sparsity = {sparsity:.1f}%",
                ha="right", va="bottom", fontsize=11,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_title(f"bandwidth = {b} pts ({band_days:.1f} d)", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

        del K_bin

        if self.save_dir is not None:
            path = os.path.join(self.save_dir, filename)
            ax.figure.savefig(path, bbox_inches="tight")
            logger.debug("Saved %s → %s", filename, path)

        if show:
            plt.show()

        return ax


