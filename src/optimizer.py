import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .data import SignalData
from .model import LinearTimeInvariantModel


class BayesianEDA:
    def __init__(
        self,
        model: LinearTimeInvariantModel,
        constraint_A=None,
        constraint_lb=np.inf,
        theta_mean=0,
        theta_variance=np.inf,
        noise_variance=1e-8,
        input_lambda=10.0,
        input_norm=0.5,
        input_threshold=0.25,
        input_density=0.05,
        input_epsilon=1e-8,
        max_iter=1000,
        max_iter_irls=10,
        tolerance_theta=0.01,
        tolerance_input=0.1,
    ):
        # State-space model
        self.model = model

        # Linear inequality constraint: A theta >= lb
        self.constraint = LinearConstraint(constraint_A, lb=constraint_lb)

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
        self.max_iter_irls = max_iter_irls
        self.tolerance_theta = tolerance_theta
        self.tolerance_input = tolerance_input

    def fit(self, theta0: np.ndarray, x0: np.ndarray, y: SignalData):
        # Initialize a new optimization
        self.initialize()

        # Initialize the parameters for input refinement
        self.initialize_input_refinement(y)

        # Initialize estimates
        theta = theta0  # Parameter estimates
        A, B, C, _ = self.model.discretize(dt=y.dt, theta=theta)  # Discretized state-space matrices
        x = np.zeros((len(y) + 1, len(x0)))  # State estimates, +1 for initial state
        x[0] = x0  # Initial state
        u = self.get_input_expectation(x, A, B)  # Input estimates
        Q = self.approximate_input_covariance(u, B)  # Input covariance

        # Expectation-Maximization (EM) algorithm
        for _ in range(self.max_iter):
            # E-step: Estimate states and inputs with iteratively reweighted least squares (IRLS)
            for _ in range(self.max_iter_irls):
                x, P = self.kalman_filter(x0, y, Q, A, C)
                x, P = self.kalman_smoother(x, P, Q, A)
                u = self.get_input_expectation(x, A, B)
                Q = self.approximate_input_covariance(u, B)

            # M-step: Optimize parameters
            theta = self.maximize_likelihood(theta, x0, y, u)
            A, B, C, _ = self.model.discretize(dt=y.dt, theta=theta)

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

    def get_input_expectation(self, x, A, B):
        # Expectation of input
        Bu = x[1:] - np.einsum("ij,kj->ki", A, x[:-1])  # Expectation of B u_k = x_{k+1} - A x_k
        u = np.einsum("ij,kj->ki", np.linalg.pinv(B), Bu)  # Least squares solution for u_k

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
        idxs = np.argsort(-u, axis=None, stable=True)[self.input_max_number :]  # Indices to remove
        u[idxs] = 0
        return u

    def approximate_input_covariance(self, u, B):
        # Approximated Gaussian covariance of Bu
        epsilon = self.input_epsilon * np.eye(B.shape[0])  # Jitter to ensure nonsingular
        Q = np.einsum("ij,kj,jl->kil", B, u ** (2 - self.input_norm) / self.input_lambda, B.T)
        return Q + epsilon

    def kalman_filter(self, x0, y, Q, A, C):
        K = len(y)  # Length of the signal
        D = len(x0)  # Dimension of the state

        # Initialize state estimate
        x = np.zeros((K + 1, D))  # +1 for initial state
        x[0] = x0

        # Initialize estimate covariance
        P = np.zeros((K + 1, D, D))  # +1 for initial state

        for k in range(K):
            # Predict
            x[k + 1] = A @ x[k]  # Predicted state estimate
            P[k + 1] = A @ P[k] @ A.T + Q[k]  # Predicted estimate covariance

            # Update
            ek = y[k] - C @ x[k + 1]  # Innovation
            Sk = C @ P[k + 1] @ C.T + self.noise_variance  # Innovation covariance
            Kk = P[k + 1] @ C.T @ np.linalg.inv(Sk)  # Optimal Kalman gain
            x[k + 1] = x[k + 1] + Kk @ ek  # Updated state estimate
            P[k + 1] = (np.eye(D) - Kk @ C) @ P[k + 1]  # Updated estimate covariance
        return x, P

    def kalman_smoother(self, x, P, Q, A):
        # Rauch-Tung-Striebel (RTS) smoother
        for k in range(len(x) - 1, 0, -1):
            xk = A @ x[k - 1]  # Predicted state estimate
            Pk = A @ P[k - 1] @ A.T + Q[k - 1]  # Predicted estimate covariance
            Ck = P[k - 1] @ A.T @ np.linalg.inv(Pk)
            x[k - 1] = x[k - 1] + Ck @ (x[k] - xk)  # Smoothed state estimate
            P[k - 1] = P[k - 1] + Ck @ (P[k] - Pk) @ Ck.T  # Smoothed estimate covariance
        return x, P

    def maximize_likelihood(self, theta, x0, y: SignalData, u):
        K = len(y)  # Length of the signal
        D = len(x0)  # Dimension of the state

        # Define objective function
        def neg_log_likelihood(_theta):
            # Get state-space matrices with new parameters
            A, B, C, _ = self.model.discretize(dt=y.dt, theta=_theta)

            # Get state estimates
            x = np.zeros((K + 1, D))  # +1 for initial state
            x[0] = x0
            for k in range(K):
                x[k + 1] = A @ x[k] + B @ u[k]
            x = x[1:]  # Remove initial state

            # Get predicted output
            Cx = np.einsum("ij,kj->ki", C, x)

            # Compute negative log-likelihood
            J = (
                np.sum((y - Cx) ** 2)
                + np.sum((_theta - self.theta_mean) ** 2 / self.theta_variance) * self.noise_variance
            )  # Scaled by 2 * noise_variance
            return J

        # Minimize with interior point method
        theta = minimize(
            neg_log_likelihood,
            theta,
            method="trust-constr",  # With inequality constraint, this becomes interior point method
            hess=lambda x: np.zeros(2 * (len(theta),)),  # delta_grad == 0 in quasi-Newton
            constraints=self.constraint,
            tol=self.tolerance_theta,
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
        theta_converged = np.allclose(theta_curr, theta_prev, rtol=0, atol=self.tolerance_theta)

        u_prev = self.inputs_log[self.iteration - 1]
        u_curr = self.inputs_log[self.iteration]
        u_converged = np.allclose(u_curr, u_prev, rtol=0, atol=self.tolerance_input)

        return theta_converged and u_converged
