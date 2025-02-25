import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.signal import find_peaks

from .data import SignalData
from .model import LinearTimeInvariantModel


class BayesianEDA:
    def __init__(
        self,
        model: LinearTimeInvariantModel,
        theta_mean=0,
        theta_variance=np.inf,
        constraints_C=None,
        constraints_b=None,
        noise_variance=1e-4**2,
        state_variance_init=1.0,
        max_iter_init=10,
        max_iter_main=1000,
        relative_tolerance_theta=0.01,
        absolute_tolerance_input=1.0,
        input_threshold_init=0.03,
        input_threshold_main=0.25,
        input_norm=0.5,
        input_epsilon=1e-5,
        input_density=0.25,
        input_prominence=0.0005,
        irls_max_iter=10,
        irls_lambda_min=1e-5,
        irls_lambda_max=1e1,
        irls_gcv_max_iter=5,
        irls_gcv_window=100,
        irls_gcv_lambda_max_iter=120,
        irls_gcv_lambda_init=1e-3,
        irls_gcv_lambda_min=1e-4,
        irls_gcv_lambda_max=1e2,
    ):
        # State-space model
        self.model = model

        # Linear inequality constraints: C theta <= b
        self.constraint = LinearConstraint(constraints_C, ub=constraints_b)

        # Hyperparameters for priors
        self.theta_mean = theta_mean
        self.theta_variance = theta_variance
        self.noise_variance = noise_variance
        self.state_variance_init = state_variance_init

        # Hyperparameters for EM
        self.max_iter_init = max_iter_init
        self.max_iter_main = max_iter_main
        self.relative_tolerance_theta = relative_tolerance_theta
        self.absolute_tolerance_input = absolute_tolerance_input

        # Hyperparameters for estimating input
        self.input_threshold_init = input_threshold_init
        self.input_threshold_main = input_threshold_main
        self.input_norm = input_norm
        self.input_epsilon = input_epsilon
        self.input_density = input_density
        self.input_prominence = input_prominence

        # Hyperparameters for IRLS
        self.irls_max_iter = irls_max_iter
        self.irls_lambda_min = irls_lambda_min
        self.irls_lambda_max = irls_lambda_max
        self.irls_gcv_max_iter = irls_gcv_max_iter
        self.irls_gcv_window = irls_gcv_window
        self.irls_gcv_lambda_max_iter = irls_gcv_lambda_max_iter
        self.irls_gcv_lambda_init = irls_gcv_lambda_init
        self.irls_gcv_lambda_min = irls_gcv_lambda_min
        self.irls_gcv_lambda_max = irls_gcv_lambda_max

    def fit(self, theta0: np.ndarray, x0: np.ndarray, y: SignalData):
        # Initialize a new optimization
        self.initialize()

        # Find peaks in the signal for input refinement
        self.find_peaks(y)

        # Initialize estimates
        theta = theta0  # Parameters
        Ad, Bd, Cd, _ = self.model.discretize(dt=y.dt, theta=theta)  # Discretized state-space matrices
        x = np.zeros((len(y) + 1, len(x0)))  # State estimates, +1 for initial state
        x[0] = x0  # Initial state
        u = self.get_input_expectation(x, Ad, Bd)  # Input estimates
        lambda_ = self.get_lambda(x, y, Cd)  # Input sparsity parameter
        Q = self.approximate_input_covariance(u, Bd, lambda_)  # Input covariance

        # Initialization phase
        for _ in range(self.max_iter_init):
            # E-step: Estimate states and inputs with IRLS
            for i in range(self.irls_max_iter):
                self.iteration_irls = i  # For input refinement
                x, P = self.kalman_filter(x0, y, Q, Ad, Cd)
                x, P = self.kalman_smoother(x, P, Q, Ad)
                lambda_ = self.get_lambda(x, y, Cd)
                u = self.get_input_expectation(x, Ad, Bd)
                Q = self.approximate_input_covariance(u, Bd, lambda_)

            # M-step: Optimize parameters
            theta = self.maximize_likelihood(theta, x, y, u, P)
            Ad, Bd, Cd, _ = self.model.discretize(dt=y.dt, theta=theta)

            self.log(theta, x, u)
            self.iteration += 1

        # Main EM phase
        for _ in range(self.max_iter_main):
            # E-step: Estimate states and inputs with IRLS
            for _ in range(self.irls_gcv_max_iter):
                x, P = self.kalman_filter(x0, y, Q, Ad, Cd)
                x, P = self.kalman_smoother(x, P, Q, Ad)
                if self.iteration < self.irls_gcv_lambda_max_iter:
                    lambda_ = self.get_lambda_gcv(x, y, u, Ad, Bd, Cd)
                u = self.get_input_expectation(x, Ad, Bd)
                Q = self.approximate_input_covariance(u, Bd, lambda_)

            # M-step: Optimize parameters
            theta = self.maximize_likelihood(theta, x, y, u, P)
            Ad, Bd, Cd, _ = self.model.discretize(dt=y.dt, theta=theta)

            self.log(theta, x, u)
            if self.check_convergence():
                break
            self.iteration += 1

        return self.theta_log, self.states_log, self.inputs_log

    def initialize(self):
        # Initialize iteration counter
        self.iteration = 0
        self.iteration_irls = 0  # For input refinement

        # Initialize log
        self.theta_log = []
        self.states_log = []
        self.inputs_log = []

    def get_input_expectation(self, x, Ad, Bd):
        # Expectation of input
        Bu = x[1:] - np.einsum("ij,kj->ki", Ad, x[:-1])  # Expectation of B u_k = x_{k+1} - A x_k
        u = np.einsum("ij,kj->ki", np.linalg.pinv(Bd), Bu)  # Least squares solution for u_k

        # Heuristic refinement
        u = self.refine_input(u)
        return u

    def approximate_input_covariance(self, u, Bd, lambda_):
        # Covariance of input
        Bu = np.einsum("ij,kj->ki", Bd, u)
        BuBuT = np.einsum("ki,kj->kij", Bu, Bu)
        epsilon = self.input_epsilon**2 * np.eye(BuBuT.shape[-1])  # Perturbation for numerical stability

        # Compute fractional power of real symmetric matrix: A^p = V D^p V^T
        D_flat, V = np.linalg.eigh(BuBuT + epsilon)  # Diagonalization of real symmetric matrix
        D = np.zeros((D_flat.shape[0], D_flat.shape[1], D_flat.shape[1]))  # Diagonal matrix with eigenvalues
        k, i = np.indices((D_flat.shape[0], D_flat.shape[1]), sparse=True)
        D[k, i, i] = D_flat
        Q = 1 / lambda_ * np.einsum("kij,kjl,klm->kim", V, D ** ((2 - self.input_norm) / 2), V.transpose(0, 2, 1))
        return Q

    def kalman_filter(self, x0, y, Q, Ad, Cd):
        K = len(y)  # Length of the signal
        D = len(x0)  # Dimension of the state

        # Initialize state estimate
        x = np.zeros((K + 1, D))  # +1 for initial state
        x[0] = x0

        # Initialize estimate covariance
        P = np.zeros((K + 1, D, D))  # +1 for initial state
        P[0] = self.state_variance_init * np.ones((D, D))

        for k in range(1, K + 1):  # +1 for initial state
            # Predict
            x[k] = Ad @ x[k - 1]  # Predicted state estimate
            P[k] = Ad @ P[k - 1] @ Ad.T + Q[k - 1]  # Predicted estimate covariance

            # Update
            vk = y[k - 1] - Cd @ x[k]  # Innovation
            Sk = Cd @ P[k] @ Cd.T + self.noise_variance  # Innovation covariance
            Kk = P[k] @ Cd.T @ np.linalg.inv(Sk)  # Optimal Kalman gain
            x[k] = x[k] + Kk @ vk  # Updated state estimate
            P[k] = (np.eye(D) - Kk @ Cd) @ P[k]  # Updated estimate covariance

        return x, P

    def kalman_smoother(self, x, P, Q, Ad):
        for k in range(len(x) - 2, -1, -1):
            # Predict
            mk = Ad @ x[k]
            Pk = Ad @ P[k] @ Ad.T + Q[k]

            # Update
            G = P[k] @ Ad.T @ np.linalg.inv(Pk)
            x[k] = x[k] + G @ (x[k + 1] - mk)
            P[k] = P[k] + G @ (P[k + 1] - Pk) @ G.T
        return x, P

    def maximize_likelihood(self, theta, x, y: SignalData, u, P):
        # Remove initial state
        x = x[1:]
        P = P[1:]

        # Define objective function
        def neg_log_likelihood(_theta):
            Ad, Bd, Cd, _ = self.model.discretize(dt=y.dt, theta=_theta)

            nll = 0
            # nll += 1 / 2 * np.linalg.vector_norm(y) ** 2  # Constant w.r.t. theta
            nll += 1 / 2 * np.trace(Ad @ (np.einsum("ki,kj->ij", x, x) + np.einsum("kij->ij", P)) @ Ad.T)
            nll -= np.trace(Ad * np.einsum("k,ij,kj->", y, Cd, x))
            nll -= np.trace(Bd @ np.einsum("k,ij,kj->ij", y, Cd, u))
            nll += np.trace(Bd @ np.einsum("ki,kj->ij", u, u) @ Bd.T)
            nll += np.trace(Ad @ np.einsum("ki,kj->ij", x, u) @ Bd.T)
            nll += np.sum((_theta - self.theta_mean) ** 2 / self.theta_variance)
            return nll.item()

        # Minimize with interior point method
        theta = minimize(
            neg_log_likelihood,
            theta,
            method="trust-constr",  # With inequality constraints, this becomes interior point method
            constraints=self.constraint,
            hess=lambda x: np.zeros(2 * (len(theta),)),  # delta_grad == 0 in quasi-Newton
        ).x
        return theta

    def log(self, theta, x, u):
        self.theta_log.append(theta)
        self.states_log.append(x)
        self.inputs_log.append(u)

        print(f"Iteration {self.iteration}: theta = {theta}")

    def check_convergence(self):
        if self.iteration == 0:
            return False

        theta_prev = self.theta_log[self.iteration - 1]
        theta_curr = self.theta_log[self.iteration]
        theta_converged = np.allclose(theta_curr, theta_prev, rtol=self.relative_tolerance_theta)

        u_prev = self.inputs_log[self.iteration - 1]
        u_curr = self.inputs_log[self.iteration]
        u_converged = np.allclose(u_curr, u_prev, atol=self.absolute_tolerance_input)

        return theta_converged and u_converged

    def find_peaks(self, y: SignalData):
        self.input_min_distance = y.f
        peaks, _ = find_peaks(y, distance=self.input_min_distance, prominence=self.input_prominence)
        self.input_desired_num = len(peaks) + np.round(y.T * self.input_density).astype(int)

    def refine_input(self, u):
        # Heuristic thresholding
        if self.iteration < 0.2 * self.max_iter_init:
            u[u < 0] = 0
        elif self.iteration < 0.4 * self.max_iter_init:
            u[u < 0.2 * self.input_threshold_init] = 0
        elif self.iteration < 0.6 * self.max_iter_init:
            u[u < 0.5 * self.input_threshold_init] = 0
        elif self.iteration < self.max_iter_init:
            u[u < self.input_threshold_init] = 0
        else:  # Main phase
            u[u < self.input_threshold_main] = 0

        # Restrict multiple input in window of minimum distance
        if self.iteration_irls > 0.5 * self.irls_max_iter:
            nonzero_idxs = np.where(u > 0)[0]
            for idx in nonzero_idxs:
                # Find the window to restrict
                left = max(0, idx - self.input_min_distance)
                right = min(len(u), idx + self.input_min_distance + 1)

                # Remove inputs that are not the maximum in the window
                max_idx = left + np.argmax(u[left:right])
                u[left:max_idx] = 0
                u[max_idx + 1 : right] = 0

        # Pick the highest desired number of inputs
        if self.iteration_irls > 0.5 * self.irls_max_iter:
            # Get the indices to remove
            idxs = np.argsort(-u, stable=True)[self.input_desired_num :]  # Stable sort to preserve order for ties
            u[idxs] = 0
        return u

    def get_lambda(self, x, y, Cd):
        x = x[1:]  # Remove initial state
        Cx = np.einsum("ij,kj->ki", Cd, x)
        Cx = np.squeeze(Cx, axis=-1)  # (K, 1) -> (K,), ensure broadcastable with y

        lambda_ = (1 - np.linalg.vector_norm(y - Cx) ** 2 / np.linalg.vector_norm(y) ** 2) * self.irls_lambda_max
        lambda_ = np.clip(lambda_, a_min=self.irls_lambda_min, a_max=None)
        return lambda_

    def get_lambda_gcv(self, x, y, u, Ad, Bd, Cd):
        M = self.irls_gcv_window
        p = self.input_norm

        # Get lambda for each window
        lambda_ = np.zeros(len(u))
        for i in range(M, len(u), M):
            # Create state-space matrices for the window
            F = np.zeros((M, Ad.shape[0]))  # [C, ..., C A^(M-1)]^T
            F[0] = np.squeeze(Cd, axis=0)
            for j in range(1, M):
                F[j] = F[j - 1] @ Ad

            AkB = np.zeros((M, Ad.shape[1]))  # [A^(M-1) B, ..., B]^T
            AkB[-1] = np.squeeze(Bd, axis=1)
            for j in range(M - 2, -1, -1):
                AkB[j] = Ad @ AkB[j + 1]
            CAkB = np.einsum("ij,kj->ki", Cd, AkB)  # [C A^(M-1) B, ..., C B]^T

            D = np.zeros((M, M))  # D[j] = [C A^(j-1) B, ..., B, 0, ..., 0]^T
            for j in range(M):
                D[j, :j] = np.squeeze(CAkB[M - j :], axis=1)

            # Define the GCV function
            Bd_norm = np.linalg.vector_norm(Bd, ord=p) ** p
            Du = y[i - M : i] - F @ x[i - M]
            W = np.diagflat(u[i - M : i] ** ((2 - p) / 2))
            U, S, _ = np.linalg.svd(D @ W)
            E = np.diagonal(U.T @ np.einsum("i,j->ij", Du, Du) @ U)

            def gcv(_lambda):
                gamma = _lambda * Bd_norm
                return np.sum(E * (gamma / (S**2 + gamma)) ** 2) / ((1 / M) * np.sum(gamma / (S**2 + gamma)) ** 2)

            # Optimize the GCV function
            _lambda = minimize(
                fun=gcv,
                x0=[self.irls_gcv_lambda_init],
                bounds=[(self.irls_gcv_lambda_min, self.irls_gcv_lambda_max)],
            ).x.item()

            lambda_[i - M : i] = _lambda  # Fill the window with the lambda
        lambda_[i:] = _lambda  # Fill the rest with the last lambda
        lambda_ = lambda_[..., None, None]  # Ensure lambda is broadcastable to compute Q
        return lambda_
