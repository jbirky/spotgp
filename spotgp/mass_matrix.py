"""Mass matrix estimation methods, mixed into GPSolver."""

import jax
import jax.numpy as jnp


class MassMatrixMixin:
    """Mass matrix estimation methods for GPSolver."""

    def _build_neg_log_lik(self, force_dense=False):
        """Return a JIT-compiled negative log-likelihood function.

        Parameters
        ----------
        force_dense : bool
            If True, always use the dense ``_gp_log_likelihood`` regardless of
            ``self.matrix_solver``.  The Hessian-based mass matrices use this
            so the curvature comes from the exact (un-truncated) covariance
            rather than the banded approximation.
        """
        from .gp_solver import _gp_log_likelihood, _gp_log_likelihood_banded

        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.harmonics, self.n_lat, self.lat_range
        fit_sn = self.fit_sigma_n
        to_phys = self._to_physical
        u_dt = self.uniform_dt
        # Kernel evaluation through the Term seam — the same closure the
        # log-posterior uses, so the curvature sees the exact kernel
        # (custom envelopes / visibility included) and the correct
        # parameter count for any layout.
        k_fn = self.kernel_sum.k_of_lag
        n_kernel = len(self.kernel_sum.param_keys)

        if self.matrix_solver == "cholesky_banded" and not force_dense:
            b = self.bandwidth
            band_tab = self._band_lag_table()

            @jax.jit
            def neg_log_lik(theta_arr):
                return -_gp_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn, b,
                    n_kernel=n_kernel,
                    uniform_dt=u_dt,
                    band_lag_table=band_tab,
                    k_of_lag=k_fn)
        else:
            full_tab = self._full_lag_table()

            @jax.jit
            def neg_log_lik(theta_arr):
                return -_gp_log_likelihood(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn,
                    n_kernel=n_kernel,
                    uniform_dt=u_dt,
                    lag_table=full_tab,
                    k_of_lag=k_fn)

        return neg_log_lik


    # =================================================================
    # Mass matrix estimation: Method 1 -- Hessian at MAP
    # =================================================================

    def mass_matrix_hessian_map(self, theta_map=None):
        """
        Estimate the inverse mass matrix from the Hessian of the
        negative log-likelihood at the MAP.

        ``M^{-1} = H^{-1}``  where  ``H = d^2(-log L)/d theta^2`` at the MAP.

        Parameters
        ----------
        theta_map : array_like, optional
            MAP estimate. If None, calls fit_map() first.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.fit_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        # Dense likelihood: curvature from the exact (un-truncated) covariance
        neg_log_lik = self._build_neg_log_lik(force_dense=True)

        hessian_fn = jax.jit(jax.hessian(neg_log_lik))
        H = jax.block_until_ready(hessian_fn(theta_map))

        # Regularize: ensure positive-definite
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, 1e-6)
        H_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(H_reg)
        self._hessian = H
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 2 -- Fisher information (analytic)
    # =================================================================


    def mass_matrix_fisher(self, theta_map=None, eigval_clip=1e-6, white_noise=1e-8):
        """
        Estimate the inverse mass matrix from the Fisher information.

        For the GP log-likelihood:

            I_{ij} = (1/2) tr(K^{-1} dK/dtheta_i  K^{-1} dK/dtheta_j)

        When ``matrix_solver="cholesky_full"``, the kernel derivatives
        dK/dtheta_i are computed via JAX forward-mode autodiff (jacfwd)
        on the full N×N covariance matrix.

        When ``matrix_solver="cholesky_banded"``, the exact Fisher requires
        the dense N×N kernel and its inverse, which would defeat the purpose
        of banded storage.  Instead, the Fisher is approximated by the
        Hessian of the banded negative log-likelihood at the MAP
        (Fisher ≈ observed information at the MLE).

        Parameters
        ----------
        theta_map : array_like, optional
            Point at which to evaluate Fisher. If None, uses MAP.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.fit_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        # Banded path: approximate Fisher via Hessian of the dense
        # log-likelihood (curvature from the exact, un-truncated covariance).
        if self.matrix_solver == "cholesky_banded":
            neg_log_lik = self._build_neg_log_lik(force_dense=True)
            hessian_fn = jax.jit(jax.hessian(neg_log_lik))
            H = jax.block_until_ready(hessian_fn(theta_map))

            eigvals, eigvecs = jnp.linalg.eigh(H)
            eigvals = jnp.maximum(eigvals, eigval_clip)
            fisher_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

            self.inverse_mass_matrix = jnp.linalg.inv(fisher_reg)
            self._fisher_matrix = H
            return self.inverse_mass_matrix

        # Dense path: exact Fisher via kernel Jacobian
        N = self.N
        n_params = theta_map.shape[0]
        if self._lag_flat is None:
            self._lag_flat = jnp.abs(
                self.x[:, None] - self.x[None, :]).ravel()
        lag_flat = self._lag_flat
        fit_sn = self.fit_sigma_n

        to_phys = self._to_physical
        k_fn = self.kernel_sum.k_of_lag
        n_kernel = len(self.kernel_sum.param_keys)

        def K_noise_flat_from_theta(theta_arr):
            """Return the full K_noise matrix as a flat vector."""
            theta_arr = to_phys(theta_arr)
            if fit_sn:
                theta_kernel = theta_arr[:n_kernel]
                sigma_n = theta_arr[n_kernel]
            else:
                theta_kernel = theta_arr
                sigma_n = 0.0

            K_flat = k_fn(theta_kernel, lag_flat)
            K = K_flat.reshape(N, N)
            noise_var = self.yerr ** 2 + sigma_n ** 2
            K_noise = K + jnp.diag(noise_var) + white_noise * jnp.eye(N)
            return K_noise.ravel()

        jacfwd_fn = jax.jit(jax.jacfwd(K_noise_flat_from_theta))
        dK_flat_dtheta = jax.block_until_ready(jacfwd_fn(theta_map))
        dK_dtheta = dK_flat_dtheta.reshape(N, N, n_params)

        K_noise_flat = K_noise_flat_from_theta(theta_map)
        K = K_noise_flat.reshape(N, N)
        K_inv = jnp.linalg.inv(K)

        K_inv_dK = jnp.einsum('ab,bcj->acj', K_inv, dK_dtheta)
        fisher = 0.5 * jnp.einsum('abi,baj->ij', K_inv_dK, K_inv_dK)

        # Regularize
        eigvals, eigvecs = jnp.linalg.eigh(fisher)
        eigvals = jnp.maximum(eigvals, eigval_clip)
        fisher_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(fisher_reg)
        self._fisher_matrix = fisher
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 3 -- Laplace approximation
    # =================================================================


    def mass_matrix_laplace(self, theta_map=None, eigval_clip=1e-6):
        """
        Laplace approximation: inverse mass matrix = inverse Hessian
        of the negative log-likelihood at the MAP.

        The posterior is approximated as:

            p(theta | data) ~ N(theta_MAP, H^{-1})

        Parameters
        ----------
        theta_map : array_like, optional
            MAP estimate. If None, calls fit_map() first.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.fit_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        neg_log_lik = self._build_neg_log_lik()

        hessian_fn = jax.jit(jax.hessian(neg_log_lik))
        H = jax.block_until_ready(hessian_fn(theta_map))

        # Regularize
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, eigval_clip)
        H_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(H_reg)
        self._laplace_hessian = H
        self._laplace_mean = theta_map
        return self.inverse_mass_matrix


    def laplace_samples(self, n_samples=1000, rng_key=None):
        """
        Draw samples from the Laplace (Gaussian) approximation
        to the posterior.

        Parameters
        ----------
        n_samples : int
        rng_key : jax.random.PRNGKey, optional

        Returns
        -------
        samples : jnp.ndarray, shape (n_samples, n_params)
        """
        if self.inverse_mass_matrix is None:
            self.mass_matrix_laplace()

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        mean = self.map_estimate
        cov = self.inverse_mass_matrix

        return jax.random.multivariate_normal(
            rng_key, mean, cov, shape=(n_samples,))


    def _get_mass_matrix(self, method, theta_ref):
        """Compute inverse mass matrix using the specified method."""
        if method is None:
            n = theta_ref.shape[0]
            self.inverse_mass_matrix = jnp.eye(n)
        elif method == "hessian_map":
            self.mass_matrix_hessian_map(theta_ref)
        elif method == "fisher":
            self.mass_matrix_fisher(theta_ref)
        elif method == "laplace":
            self.mass_matrix_laplace(theta_ref)
        elif method == "diagonal":
            hessian_fn = jax.jit(jax.hessian(self.neg_log_posterior))
            H = jax.block_until_ready(hessian_fn(theta_ref))
            diag = jnp.maximum(jnp.diag(H), 1e-6)
            self.inverse_mass_matrix = jnp.diag(1.0 / diag)
        else:
            raise ValueError(f"Unknown mass_matrix_method: {method}")

        return self.inverse_mass_matrix

