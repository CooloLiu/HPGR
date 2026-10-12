"""Hierarchical probabilistic granular-ball system (HPGR) -- implementation of Section 4.

Organized by paper subsection:

    Section 4.2 Probabilistic granular-ball representation -> initialize_single_ball
    Section 4.3 Localized structural likelihood principle  -> should_split_ball
    Section 4.4 Hierarchical construction                  -> split_ball,
                                                              adaptive_split_evolution,
                                                              assign_data_to_balls
    Section 4.5 Hierarchical anomaly scoring               -> compute_anomaly_scores,
                                                              _compute_ball_distribution_entropy,
                                                              _normalize_ball_entropy_weights

The mixture density and the entropy-based level weights live in scoring.py.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

from . import scoring
from .granular_ball import GranularBall


class ProbabilisticGranularBallSystem:

    def __init__(self,
                 bic_threshold=10.0,
                 tau_ll=10.0,
                 use_ball_entropy_weighting=False,
                 random_state=None):
        self.balls = []
        self.bic_threshold = bic_threshold
        self.tau_ll = tau_ll
        self.ball_id_counter = 0
        self.use_ball_entropy_weighting = use_ball_entropy_weighting  # True: entropy-weighted accumulation, False: final level only
        self.random_state = random_state
        self.layer_weights = []  # per-level weights

        # Set global randomness.
        if self.random_state is not None:
            import random
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    # ---------------- Section 4.2 Probabilistic granular-ball representation ----------------
    def initialize_single_ball(self, data):
        center = np.mean(data, axis=0)
        covariance = np.cov(data.T)
        if np.linalg.det(covariance) < 1e-10:
            covariance += np.eye(covariance.shape[0]) * 1e-6
        ball = GranularBall(center, covariance, ball_id=self.ball_id_counter)
        ball.data_points = data.copy()
        self.balls = [ball]
        self.ball_id_counter += 1

    # ---------------- Section 4.3 Localized structural likelihood principle ----------------
    def should_split_ball(self, ball):
        """Test a region locally: split only if the BIC drop and the likelihood gain both pass."""
        try:
            gmm_1 = GaussianMixture(n_components=1, random_state=self.random_state, n_init=10)
            gmm_1.fit(ball.data_points)
            bic_1 = gmm_1.bic(ball.data_points)

            gmm_2 = GaussianMixture(n_components=2, random_state=self.random_state, n_init=10)
            gmm_2.fit(ball.data_points)
            bic_2 = gmm_2.bic(ball.data_points)

            log_likelihood_original = gmm_1.score_samples(ball.data_points)
            log_likelihood_split = gmm_2.score_samples(ball.data_points)
            delta_lk = np.sum(log_likelihood_split - log_likelihood_original)

            if (bic_2 < bic_1 - self.bic_threshold) and (delta_lk > self.tau_ll):
                return True, gmm_2
            return False, None
        except Exception:
            return False, None

    # ---------------- Section 4.4 Hierarchical construction ----------------
    def split_ball(self, ball, gmm_2_model):
        labels = gmm_2_model.predict(ball.data_points)
        new_balls = []
        for i in range(2):
            cluster_data = ball.data_points[labels == i]
            center = gmm_2_model.means_[i]
            covariance = gmm_2_model.covariances_[i]
            if np.linalg.det(covariance) < 1e-10:
                covariance += np.eye(covariance.shape[0]) * 1e-6
            new_ball = GranularBall(center, covariance, self.ball_id_counter)
            new_ball.data_points = cluster_data
            new_balls.append(new_ball)
            self.ball_id_counter += 1
        return new_balls

    def adaptive_split_evolution(self, data, max_iterations=20, eval_data=None):
        self.initialize_single_ball(data)
        non_splittable_balls = set()
        all_anomaly_scores = []
        self.layer_weights = []

        if eval_data is None:
            eval_data = data

        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")

            # Per-level scores and entropy weights are only needed for entropy-weighted accumulation.
            if self.use_ball_entropy_weighting:
                scores = self.compute_anomaly_scores(eval_data)
                min_score = np.min(scores)
                max_score = np.max(scores)
                if max_score > min_score:
                    norm_scores = (scores - min_score) / (max_score - min_score)
                else:
                    norm_scores = scores
                all_anomaly_scores.append(norm_scores)
                self.layer_weights.append(self._compute_ball_distribution_entropy())

            new_balls = []
            split_occurred = False
            for ball in self.balls:
                if ball.ball_id in non_splittable_balls:
                    new_balls.append(ball)
                    continue
                should_split, gmm_2 = self.should_split_ball(ball)
                if should_split:
                    new_balls.extend(self.split_ball(ball, gmm_2))
                    split_occurred = True
                else:
                    non_splittable_balls.add(ball.ball_id)
                    new_balls.append(ball)
            self.balls = new_balls

            print(f"  Balls after iteration {iteration + 1}: {len(self.balls)}")

            if not split_occurred:
                print("  No split occurred, stopping.")
                break

        print("\nEvolution finished, assigning all samples to the final balls...")
        self.assign_data_to_balls(data)

        if self.use_ball_entropy_weighting:
            final_scores = self.compute_anomaly_scores(eval_data)
            min_score = np.min(final_scores)
            max_score = np.max(final_scores)
            if max_score > min_score:
                norm_scores = (final_scores - min_score) / (max_score - min_score)
            else:
                norm_scores = final_scores
            all_anomaly_scores.append(norm_scores)

            self.layer_weights.append(self._compute_ball_distribution_entropy())
            print(f"  Final-level ball-distribution entropy: {self.layer_weights[-1]:.4f}")

            self._normalize_ball_entropy_weights()
            print(f"  Normalized level weights: {[f'{w:.4f}' for w in self.layer_weights]}")
            return self.balls, all_anomaly_scores, self.layer_weights

        print("Using the final-level anomaly scores only (no accumulation).")
        return self.balls, [], []

    def assign_data_to_balls(self, data):
        """Assign every sample to the ball with the highest likelihood."""
        for ball in self.balls:
            ball.data_points = []
        for point in data:
            best_ball = None
            best_likelihood = -np.inf
            for ball in self.balls:
                likelihood = ball.log_likelihood(point)
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_ball = ball
            if best_ball is not None:
                best_ball.data_points.append(point)
        for ball in self.balls:
            ball.data_points = np.array(ball.data_points) if ball.data_points else np.empty((0, data.shape[1]))

    # ---------------- Section 4.5 Hierarchical anomaly scoring ----------------
    def compute_anomaly_scores(self, data):
        """Section 4.5: the current level forms a Gaussian mixture; score = -log p(x)."""
        if not self.balls:
            return np.full(len(data), np.inf)

        valid_balls = [ball for ball in self.balls if len(ball.data_points) > 0]
        if not valid_balls:
            return np.full(len(data), np.inf)

        weights = np.array([len(ball.data_points) for ball in valid_balls], dtype=float)
        weights = np.maximum(weights / np.sum(weights), 1e-8)
        weights = weights / np.sum(weights)

        means = np.array([ball.center for ball in valid_balls])
        covariances = np.array([scoring.regularize_covariance(ball.covariance)
                                for ball in valid_balls])

        return scoring.mixture_anomaly_scores(data, means, covariances, weights)

    def _normalize_ball_entropy_weights(self):
        """Section 4.5: turn per-level ball-distribution entropies into normalized level weights."""
        if not self.layer_weights:
            return
        print(f"  Raw ball-distribution entropies: {[f'{e:.4f}' for e in self.layer_weights]}")
        self.layer_weights = list(scoring.normalize_entropy_weights(self.layer_weights))

    def _compute_ball_distribution_entropy(self):
        """Section 4.5: entropy of the ball-mass distribution at the current level."""
        counts = [len(ball.data_points) for ball in self.balls if len(ball.data_points) > 0]
        return scoring.distribution_entropy(counts)
