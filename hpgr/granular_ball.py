"""A single probabilistic granular-ball B_k := N(mu_k, Sigma_k) (Section 4.2)."""
import numpy as np


class GranularBall:

    def __init__(self, center, covariance, ball_id=0):
        self.center = center            # mu_k
        self.covariance = covariance    # Sigma_k
        self.ball_id = ball_id
        self.data_points = []           # samples assigned to this ball

        self._cache = {}
        self._cache_valid = False
        self._update_cache()

    def _update_cache(self):
        """Cache the inverse and the determinant of the covariance matrix."""
        try:
            # Numerical stability: keep the covariance positive definite.
            min_eigenval = np.min(np.linalg.eigvalsh(self.covariance))
            if min_eigenval < 1e-8:
                reg_factor = max(1e-6 - min_eigenval, 1e-6)
                self.covariance += np.eye(self.covariance.shape[0]) * reg_factor

            self._cache['inv_covariance'] = np.linalg.inv(self.covariance)
            self._cache['det_covariance'] = np.linalg.det(self.covariance)

            # Keep the determinant positive.
            if self._cache['det_covariance'] <= 0:
                self._cache['det_covariance'] = 1e-6

            self._cache_valid = True
        except np.linalg.LinAlgError:
            # Handle a singular covariance.
            self.covariance += np.eye(self.covariance.shape[0]) * 1e-6
            self._update_cache()

    def log_likelihood(self, point):
        """Log-likelihood of a single point under N(mu_k, Sigma_k)."""
        if not self._cache_valid:
            self._update_cache()

        diff = point - self.center
        try:
            mahalanobis = np.dot(np.dot(diff.T, self._cache['inv_covariance']), diff)
            det_cov = max(self._cache['det_covariance'], 1e-8)
            return -0.5 * (mahalanobis + np.log(det_cov) + len(point) * np.log(2 * np.pi))
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
