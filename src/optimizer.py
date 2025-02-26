import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .data import SignalData
from .model import LinearTimeInvariantModel


class BayesianEDA:
    def __init__(
        self,
        model: LinearTimeInvariantModel,
        constraints_C=None,
        constraints_b=np.inf,
        theta_mean=0,
        theta_variance=np.inf,
        noise_variance=1e-4**2,
        input_lambda=10.0,
        input_norm=0.5,
        input_threshold=0.25,
        input_density=0.05,
        input_epsilon=1e-8,
        max_iter=1000,
        irls_max_iter=10,
        relative_tolerance_theta=0.01,
        absolute_tolerance_input=0.1,
    ):
        # State-space model
        self.model = model

        # Linear inequality constraints: C theta <= b
        self.constraint = LinearConstraint(constraints_C, ub=constraints_b)

        # Hyperparameters for priors
        self.theta_mean = theta_mean
        self.theta_variance = theta_variance
        self.noise_variance = noise_variance

        # Hyperparameters for estimating input
        self.input_lambda = input_lambda
        self.input_norm = input_norm
        self.input_threshold = input_threshold
        self.input_density = input_density
        self.input_epsilon = input_epsilon

        # Hyperparameters for EM algorithm
        self.max_iter = max_iter
        self.irls_max_iter = irls_max_iter
        self.relative_tolerance_theta = relative_tolerance_theta
        self.absolute_tolerance_input = absolute_tolerance_input

    def fit(self, theta0: np.ndarray, x0: np.ndarray, y: SignalData):
        # Initialize a new optimization
        self.initialize()

        # Initialize the parameters for input refinement
        self.initialize_input_refinement(y)

        # Initialize estimates
        theta = theta0  # Parameter estimates
        Ad, Bd, Cd, _ = self.model.discretize(dt=y.dt, theta=theta)  # Discretized state-space matrices
        x = np.zeros((len(y) + 1, len(x0)))  # State estimates, +1 for initial state
        x[0] = x0  # Initial state
        u = self.get_input_expectation(x, Ad, Bd)  # Input estimates
        Q = self.approximate_input_covariance(u, Bd)  # Input covariance

        # Expectation-Maximization (EM) algorithm
        for _ in range(self.max_iter):
            # E-step: Estimate states and inputs with iteratively reweighted least squares (IRLS)
            for _ in range(self.irls_max_iter):
                x, P = self.kalman_filter(x0, y, Q, Ad, Cd)
                x, P = self.kalman_smoother(x, P, Q, Ad)
                u = self.get_input_expectation(x, Ad, Bd)
                Q = self.approximate_input_covariance(u, Bd)

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

        # Initialize log
        self.theta_log = []
        self.states_log = []
        self.inputs_log = []

    def initialize_input_refinement(self, y: SignalData):
        self.input_min_interval = y.f  # At most one input per second
        self.input_max_number = np.round(y.T * self.input_density).astype(int)

    def get_input_expectation(self, x, Ad, Bd):
        # Expectation of input
        Bu = x[1:] - np.einsum("ij,kj->ki", Ad, x[:-1])  # Expectation of B u_k = x_{k+1} - A x_k
        u = np.einsum("ij,kj->ki", np.linalg.pinv(Bd), Bu)  # Least squares solution for u_k

        # Heuristic refinement
        u = self.refine_input(u)
        return u

    def refine_input(self, u):
        # Thresholding
        u[u < self.input_threshold] = 0

        # Restrict multiple inputs in window of minimum interval
        nonzero_idxs = np.where(u > 0)[0]
        for idx in nonzero_idxs:
            # Find the window to restrict
            left = max(0, idx - self.input_min_interval)
            right = min(len(u), idx + self.input_min_interval + 1)

            # Remove inputs that are not the maximum in the window
            max_idx = left + np.argmax(u[left:right])
            u[left:max_idx] = 0
            u[max_idx + 1 : right] = 0

        # Pick the top max number of inputs
        idxs = np.argsort(-u, stable=True)[self.input_max_number :]  # Indices to remove
        u[idxs] = 0
        return u

    def approximate_input_covariance(self, u, Bd):
        # Covariance of input
        Bu = np.einsum("ij,kj->ki", Bd, u)
        BuBuT = np.einsum("ki,kj->kij", Bu, Bu)
        epsilon = self.input_epsilon * np.eye(BuBuT.shape[-1])  # Perturbation for numerical stability

        # Compute fractional power of real symmetric matrix: A^p = V D^p V^T
        D_flat, V = np.linalg.eigh(BuBuT + epsilon)  # Diagonalization of real symmetric matrix
        D = np.zeros((D_flat.shape[0], D_flat.shape[1], D_flat.shape[1]))  # Diagonal matrix with eigenvalues
        k, i = np.indices((D_flat.shape[0], D_flat.shape[1]), sparse=True)
        D[k, i, i] = D_flat
        Q = np.einsum("kij,kjl,klm->kim", V, D ** ((2 - self.input_norm) / 2), V.transpose(0, 2, 1)) / self.input_lambda
        return Q

    def kalman_filter(self, x0, y, Q, Ad, Cd):
        K = len(y)  # Length of the signal
        D = len(x0)  # Dimension of the state

        # Initialize state estimate
        x = np.zeros((K + 1, D))  # +1 for initial state
        x[0] = x0

        # Initialize estimate covariance
        P = np.zeros((K + 1, D, D))  # +1 for initial state

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
        self.states_log.append(x[1:])  # Remove initial state
        self.inputs_log.append(u)

        print(f"Iteration {self.iteration}: theta = {theta}")

    def check_convergence(self):
        if self.iteration == 0:
            return False

        theta_prev = self.theta_log[self.iteration - 1]
        theta_curr = self.theta_log[self.iteration]
        theta_converged = np.allclose(theta_curr, theta_prev, rtol=self.relative_tolerance_theta, atol=0)

        u_prev = self.inputs_log[self.iteration - 1]
        u_curr = self.inputs_log[self.iteration]
        u_converged = np.allclose(u_curr, u_prev, rtol=0, atol=self.absolute_tolerance_input)

        return theta_converged and u_converged
