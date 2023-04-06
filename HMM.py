import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from pandas import DataFrame as DF


def had_prod(A, B):
    """Compute the Hadamard product of two matrices."""
    return np.multiply(A, B)
    # return scipy.special.logsumexp(np.log(A) + np.log(B))

    # avoid zero values in A or B
    # np.nan_to_num(A)
    # np.nan_to_num(B)
    # A[A == 0] = 1e-10
    # B[B == 0] = 1e-10

    # return np.exp(np.log(A) + np.log(B))


def plot_heatmap(
    data, title, xLabel, yLabel, annot=False, fmt=".1f", cmap="viridis", show=True
):
    """Plot a heatmap of the data with the given title and axis labels using seaborn."""
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if show:
        plt.show()
    return plt


class HMM(object):
    """
    A class for implementing HMMs.

    Attributes
    ----------
    envShape : list
        A two element list specifying the shape of the environment
    states : list
        A list of states specified by their (x, y) coordinates
    observations : list
        A list specifying the sequence of observations
    transition_probs : numpy.ndarray
        An N x N array encoding the transition probabilities, where
        T[i,j] is the probability of transitioning from state i to state j.
        N is the total number of states (envShape[0]*envShape[1])
    emission_probs : numpy.ndarray
        An M x N array encoding the emission probabilities, where
        M[k,i] is the probability of observing k from state i.
    start_probs : numpy.ndarray
        An N x 1 array encoding the prior probabilities

    Methods
    -------
    train(observations)
        Estimates the HMM parameters using a set of observation sequences
    viterbi(observations)
        Implements the Viterbi algorithm on a given observation sequence
    setParams(T, M, pi)
        Sets the transition (T), emission (M), and prior (pi) distributions
    getParams
        Queries the transition (T), emission (M), and prior (pi) distributions
    sub2ind(i, j)
        Convert integer (i,j) coordinates to linear index.
    """

    def __init__(
        self,
        envShape,
        transition_probs=None,
        emission_probs=None,
        start_probs=None,
    ) -> None:
        """
        Initialize the class.

        Attributes
        ----------
        envShape : list
            A two element list specifying the shape of the environment
        transition_probs : numpy.ndarray, optional
            An N x N array encoding the transition probabilities, where
            transition_probs[i,j] is the probability of transitioning from state i to state j.
            N is the total number of states (envShape[0]*envShape[1])
        emission_probs : numpy.ndarray, optional
            An M x N array encoding the emission probabilities, where
            emission_probs[k,i] is the probability of observing k from state i.
        prior_pi : numpy.ndarray, optional
            An N x 1 array encoding the prior probabilities
        """
        # assert len(envShape) == 2
        # assert envShape[0] > 0
        # assert envShape[1] > 0

        self.envShape = envShape
        self.num_states = envShape[0] * envShape[1]
        self.states = [i for i in range(envShape[0] * envShape[1])]
        # print(self.num_states)
        # print(len(self.states))
        # print(self.states)
        assert len(self.states) == self.num_states

        if transition_probs is None:
            self._standard_transition_probs()
        else:
            self.transition_probs = transition_probs
        if emission_probs is None:
            self._standard_emission_probs()
        else:
            self.emission_probs = emission_probs
        if start_probs is None:
            self._standard_start_probs()
        else:
            self.start_probs = start_probs
        print(self.emission_probs)
        print(self.emission_probs.shape)
        # assert self.num_states == self.emission_probs.shape[1]
        self.state_map = self._generate_state_map(self.states)

        return None

    def setParams(self, transition_probs, emission_probs, prior_pi) -> None:
        """Set the transition, emission, and prior probabilities."""
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
        self.prior_pi = prior_pi
        return None

    def getParams(self):
        """Get the transition, emission, and prior probabilities."""
        return (self.transition_probs, self.emission_probs, self.prior_pi)

    def train(self, observations, train_iterations=10000):
        """
        Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence

        train_iterations : int
            Number of iterations to run the Baum-Welch algorithm
        """
        print("base")
        print(f"start probs: {DF(self.start_probs)}")
        print(f"emissions: {DF(self.emission_probs)}")
        # plot_heatmap(
        #     self.transition_probs,
        #     "transition_probs",
        #     "transition_probs",
        #     "transition_probs",
        #     True,
        # )
        # plot_heatmap(
        #     self.emission_np, sum(probs,
        #     "emission_probs",
        #     "emission_probs",
        #     "emission_probs",
        #     True,
        # )
        print()

        last_prob = float("inf")

        # Train the model 'iteration' number of times
        # store emission_probs and transition_probs copies since you should use same values for one loop
        for iter in range(train_iterations):
            new_emission_probs = np.asmatrix(np.zeros((self.emission_probs.shape)))
            new_transition_prob = np.asmatrix(np.zeros((self.transition_probs.shape)))
            new_start_probs = np.asmatrix(np.zeros((self.start_probs.shape)))

            for observation in observations:
                new_start_probs += self._train_start_probs(observation).T
                new_emission_probs += self._train_emission(observation)
                new_transition_prob += self._train_transition(observation)

            # Normalizing
            new_start_probs = new_start_probs / np.sum(new_start_probs, axis=1)
            new_transition_prob = new_transition_prob / np.sum(
                new_transition_prob, axis=1
            )
            new_emission_probs = new_emission_probs / np.sum(new_emission_probs, axis=1)

            self.start_probs, self.emission_probs, self.transition_probs = (
                new_start_probs,
                new_emission_probs,
                new_transition_prob,
            )

            print(f"iter: {iter}")
            print(f"start probs: {DF(self.start_probs)}")
            print(f"transitions: {DF(self.transition_probs)}")
            print(f"emissions: {DF(self.emission_probs)}")
            # plot_heatmap(
            #     self.transition_probs,
            #     "transition_probs",
            #     "transition_probs",
            #     "transition_probs",
            #     True,
            # )
            # plot_heatmap(
            #     self.emission_probs,
            #     "emission_probs",
            #     "emission_probs",
            #     "emission_probs",
            #     True,
            # )
            print()

            c_scales = [
                np.sum(np.log(self._compute_alpha(obs)[1])) for obs in observations
            ]
            curr_prob = 0
            for c_scale in c_scales:
                curr_prob -= c_scale
            if (last_prob - curr_prob) > 0.0000001:
                last_prob = curr_prob
            else:
                break

        # print(self.start_probs)
        # print("hi3")
        # print(self.emission_probs)
        return self.emission_probs, self.transition_probs, self.start_probs

    def viterbi(self, observations):
        """
        Implement the Viterbi algorithm.

        Parameters
        ----------
        observations : list
            A list specifying the sequence of observations, where each o
            observation is a string (e.g., 'r')

        Returns
        -------
        states : list
            List of predicted sequence of states, each specified as (x, y) pair
        """
        print("viterbi")
        print(f"start probs: {self.start_probs}")
        print(f"emission probs: {self.emission_probs}")
        obs_map = self._generate_obs_map(observations)

        # Find total states,observations
        num_observations = len(observations)
        num_states = self.num_states

        # initialize data
        # Path stores the state sequence giving maximum probability
        old_path = np.zeros((num_observations, num_states))
        new_path = np.zeros((num_observations, num_states))

        # Find initial delta
        # Map observation to an index
        # delta[s] stores the probability of most probable path ending in state 's'
        ob_ind = obs_map[0]
        self.emission_probs[ob_ind, :]
        delta = had_prod(np.transpose(self.emission_probs[ob_ind, :]), self.start_probs)
        print(delta)
        delta /= np.nansum(delta)
        delta = np.nan_to_num(delta)

        print(delta)

        # initialize path
        old_path[0, :] = [i for i in range(num_states)]

        # Find delta[t][x] for each state 'x' at the iteration 't'
        # delta[t][x] can be found using delta[t-1][x] and taking the maximum possible path
        for curr_t in range(1, num_observations):

            # Map observation to an index
            ob_ind = obs_map[curr_t]
            # Find temp and take max along each row to get delta
            temp = had_prod(delta, self.transition_probs.T)
            temp = had_prod(temp, self.emission_probs[ob_ind, :])

            # Update delta and scale it
            delta = temp.max(axis=1).T / np.sum(delta)
            np.nan_to_num(delta)

            # Find state which is most probable using argax
            # Convert to a list for easier processing
            max_temp = np.ravel(temp.argmax(axis=1).T).tolist()
            np.nan_to_num(max_temp)

            # Update path
            for s in range(num_states):
                new_path[:curr_t, s] = old_path[0:curr_t, max_temp[s]]

            new_path[curr_t, :] = [i for i in range(num_states)]
            old_path = new_path.copy()

        # Find the state in last stage, giving maximum probability
        final_max = np.argmax(np.ravel(delta))
        best_path = old_path[:, final_max].tolist()
        best_path_map = [self.state_map[i] for i in best_path]

        print(self.state_map)

        best_tups = [self.ind2sub(i) for i in best_path_map]
        return best_tups

    def getLogStartProb(self, state):
        """Return the log probability of a particular state."""
        return np.log(self.prior_pi[state])

    def getLogTransProb(self, fromState, toState):
        """Return the log probability associated with a state transition."""
        return np.log(self.transition_probs[toState, fromState])

    def getLogOutputProb(self, state, output):
        """Return the log probability of a state-dependent observation."""
        return np.log(self.emission_probs[output, state])

    def sub2ind(self, i, j):
        """Convert subscript (i,j) to linear index."""
        return self.envShape[1] * i + j

    def ind2sub(self, ind):
        """Convert linear index to subscript (i,j)."""
        return (ind // self.envShape[1], ind % self.envShape[1])

    def obs2ind(self, obs) -> int:
        """Convert observation string to linear index."""
        obsToInt = {"r": 0, "g": 1, "b": 2, "y": 3}
        return obsToInt[obs]

    def _generate_state_map(self, states):
        state_map = {}
        for i, o in enumerate(states):
            state_map[i] = o
        return state_map

    def _generate_obs_map(self, observations):
        return [self.obs2ind(obs) for obs in observations]

    def _compute_alpha(self, observations):
        # Calculate alpha matrix and return it

        obs_map = self._generate_obs_map(observations)
        num_observations = len(observations)

        # Initialize values
        alpha = np.asmatrix(np.zeros((self.num_states, num_observations)))
        c_scale = np.asmatrix(np.zeros((num_observations, 1)))

        # Handle alpha base case
        # alpha[:, 0] = had_prod(self.start_probs, (self.emission_probs[ob_ind, :])).T
        alpha[:, 0] = had_prod(
            self.start_probs.T, (self.emission_probs[obs_map[0], :])
        ).T

        # store scaling factors, scale alpha
        c_scale[0, 0] = 1 / np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] * c_scale[0]
        # Iteratively calculate alpha(t) for all 't'
        for curr_t in range(1, num_observations):
            alpha[:, curr_t] = np.sum(
                had_prod(alpha[:, curr_t - 1].T, self.transition_probs)
            ).T
            alpha[:, curr_t] = had_prod(
                alpha[:, curr_t].T,
                np.reshape(self.emission_probs[obs_map[curr_t], :], (1, 16)),
            ).T
            # Store scaling factors, scale alpha
            c_scale[curr_t] = 1 / np.sum(alpha[:, curr_t])
            alpha[:, curr_t] = had_prod(alpha[:, curr_t], c_scale[curr_t])

        # return the computed alpha
        return (alpha, c_scale)

    def _compute_beta(self, observations, c_scale):
        # Calculate Beta maxtrix
        obs_map = self._generate_obs_map(observations)
        num_states = self.num_states
        num_observations = len(observations)

        # Initialize values
        ob_ind = obs_map[num_observations - 1]
        beta = np.asmatrix(np.zeros((num_states, num_observations)))

        # Handle beta base case
        beta[:, num_observations - 1] = 1  # c_scale[num_observations - 1]

        # Iteratively calculate beta(t) for all 't'
        for curr_t in range(num_observations - 1, 0, -1):
            ob_ind = obs_map[curr_t]
            beta[:, curr_t - 1] = had_prod(
                beta[:, curr_t].T, self.emission_probs[ob_ind, :]
            ).T
            beta[:, curr_t - 1] = np.sum(
                had_prod(self.transition_probs, beta[:, curr_t - 1])
            )
            beta[:, curr_t - 1] = had_prod(beta[:, curr_t - 1], c_scale[curr_t - 1])

        # return the computed beta
        return beta

    def _compute_gamma(self, observations):
        # Find alpha and beta values
        alpha, c_scale = self._compute_alpha(observations)
        beta = self._compute_beta(observations, c_scale)

        alphabeta = had_prod(alpha, beta)

        # Calculate gamma
        # gamma is simply product of alpha and beta
        gamma = alphabeta / np.sum(alphabeta, axis=0)
        return gamma

    def _train_emission(self, observations):
        # Initialize matrix
        new_emission_probs = np.asmatrix(np.zeros(self.emission_probs.shape))

        # obs_map = self._generate_obs_map(observations)
        onehots = np.zeros((len(observations), self.emission_probs.shape[0]))
        for i, observation in enumerate(observations):
            onehots[i, :] = self.obs2onehot(observation)

        gamma = self._compute_gamma(observations)

        numerator = np.matmul(gamma, onehots)
        denominator = np.sum(gamma, axis=1)
        new_emission_probs = np.divide(numerator, denominator).T

        return new_emission_probs

    def ind2onehot(self, ind):
        onehot = np.zeros(self.emission_probs.shape[0])
        onehot[ind] = 1
        return onehot

    def obs2onehot(self, obs):
        onehot = np.zeros(self.emission_probs.shape[0])
        onehot[self.obs2ind(obs)] = 1
        return onehot

    def _train_transition(self, observations):
        gamma = self._compute_gamma(observations)
        obs_map = self._generate_obs_map(observations)

        # Initialize transition matrix
        new_transition_probs = np.asmatrix(np.zeros(self.transition_probs.shape))

        # Find alpha and beta
        alpha, c_scale = self._compute_alpha(observations)
        beta = self._compute_beta(observations, c_scale)

        # calculate transition matrix values
        for t in range(len(observations) - 1):
            temp = had_prod(
                self.transition_probs, had_prod(alpha[:, t], beta[:, t + 1].T)
            )
            new_transition_probs += had_prod(
                temp, self.emission_probs[obs_map[t + 1], :]
            )

        gamma_sum = np.sum(gamma, axis=1).reshape(1, self.num_states)
        new_transition_probs = np.divide(new_transition_probs, gamma_sum)

        # # Normalize values so that sum of probabilities is 1
        # for i in range(self.transition_probs.shape[0]):
        #     new_transition_probs[i, :] /= np.sum(new_transition_probs[i, :])

        return new_transition_probs

    def _train_start_probs(self, observations):
        gamma = self._compute_gamma(observations)

        return gamma[:, 0].T / sum(gamma[:, 0])

    def _standard_transition_probs(self):
        # Initial estimate of the transition function
        # where transition_probs[sub2ind(i',j'), sub2ind(i,j)] is the likelihood
        # of transitioning from (i,j) --> (i',j')
        self.transition_probs = np.zeros((self.num_states, self.num_states))

        # Self-transitions
        for i in range(self.num_states):
            self.transition_probs[i, i] = 0.2
        # Black rooms
        self.transition_probs[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
        self.transition_probs[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
        self.transition_probs[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
        self.transition_probs[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0
        # (1, 0) -->
        self.transition_probs[self.sub2ind(2, 0), self.sub2ind(1, 0)] = 0.8
        # (2, 0) -->
        self.transition_probs[self.sub2ind(1, 0), self.sub2ind(2, 0)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(2, 1), self.sub2ind(2, 0)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(3, 0), self.sub2ind(2, 0)] = 0.8 / 3.0
        # (3, 0) -->
        self.transition_probs[self.sub2ind(2, 0), self.sub2ind(3, 0)] = 0.8 / 2.0
        self.transition_probs[self.sub2ind(3, 1), self.sub2ind(3, 0)] = 0.8 / 2.0
        # (0, 1) --> (0, 2)
        self.transition_probs[self.sub2ind(0, 2), self.sub2ind(0, 1)] = 0.8
        # (2, 1) -->
        self.transition_probs[self.sub2ind(2, 0), self.sub2ind(2, 1)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(3, 1), self.sub2ind(2, 1)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(2, 2), self.sub2ind(2, 1)] = 0.8 / 3.0
        # (3, 1) -->
        self.transition_probs[self.sub2ind(2, 1), self.sub2ind(3, 1)] = 0.8 / 2.0
        self.transition_probs[self.sub2ind(3, 0), self.sub2ind(3, 1)] = 0.8 / 2.0
        # (0, 2) -->
        self.transition_probs[self.sub2ind(0, 1), self.sub2ind(0, 2)] = 0.8 / 2.0
        self.transition_probs[self.sub2ind(1, 2), self.sub2ind(0, 2)] = 0.8 / 2.0
        # (1, 2) -->
        self.transition_probs[self.sub2ind(0, 2), self.sub2ind(1, 2)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(2, 2), self.sub2ind(1, 2)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(1, 3), self.sub2ind(1, 2)] = 0.8 / 3.0
        # (2, 2) -->
        self.transition_probs[self.sub2ind(1, 2), self.sub2ind(2, 2)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(2, 1), self.sub2ind(2, 2)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(2, 3), self.sub2ind(2, 2)] = 0.8 / 3.0
        # (1, 3) -->
        self.transition_probs[self.sub2ind(1, 2), self.sub2ind(1, 3)] = 0.8 / 2.0
        self.transition_probs[self.sub2ind(2, 3), self.sub2ind(1, 3)] = 0.8 / 2.0
        # (2, 3) -->
        self.transition_probs[self.sub2ind(1, 3), self.sub2ind(2, 3)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(3, 3), self.sub2ind(2, 3)] = 0.8 / 3.0
        self.transition_probs[self.sub2ind(2, 2), self.sub2ind(2, 3)] = 0.8 / 3.0
        # (3, 3) --> (2, 3)
        self.transition_probs[self.sub2ind(2, 3), self.sub2ind(3, 3)] = 0.8
        return None

    def _standard_emission_probs(self):
        # Initial estimates of emission likelihoods, where
        # emission_probs[k, sub2ind(i,j)]: likelihood of observation k from state (i, j)
        self.emission_probs = np.ones((4, 16)) * 0.1

        # Black states
        self.emission_probs[:, self.sub2ind(0, 0)] = 0.25
        self.emission_probs[:, self.sub2ind(1, 1)] = 0.25
        self.emission_probs[:, self.sub2ind(0, 3)] = 0.25
        self.emission_probs[:, self.sub2ind(3, 2)] = 0.25

        self.emission_probs[self.obs2ind("r"), self.sub2ind(0, 1)] = 0.7
        self.emission_probs[self.obs2ind("g"), self.sub2ind(0, 2)] = 0.7
        self.emission_probs[self.obs2ind("g"), self.sub2ind(1, 0)] = 0.7
        self.emission_probs[self.obs2ind("b"), self.sub2ind(1, 2)] = 0.7
        self.emission_probs[self.obs2ind("r"), self.sub2ind(1, 3)] = 0.7
        self.emission_probs[self.obs2ind("y"), self.sub2ind(2, 0)] = 0.7
        self.emission_probs[self.obs2ind("g"), self.sub2ind(2, 1)] = 0.7
        self.emission_probs[self.obs2ind("r"), self.sub2ind(2, 2)] = 0.7
        self.emission_probs[self.obs2ind("y"), self.sub2ind(2, 3)] = 0.7
        self.emission_probs[self.obs2ind("b"), self.sub2ind(3, 0)] = 0.7
        self.emission_probs[self.obs2ind("y"), self.sub2ind(3, 1)] = 0.7
        self.emission_probs[self.obs2ind("b"), self.sub2ind(3, 3)] = 0.7
        return None

    def _standard_start_probs(self):
        # Initialize estimates of prior probabilities where
        # start_probs[(i, j)] is the likelihood of starting in state (i, j)
        self.start_probs = np.ones((16, 1)) / 12
        self.start_probs[self.sub2ind(0, 0)] = 0.0
        self.start_probs[self.sub2ind(1, 1)] = 0.0
        self.start_probs[self.sub2ind(0, 3)] = 0.0
        self.start_probs[self.sub2ind(3, 2)] = 0.0
        return None
