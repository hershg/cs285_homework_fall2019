import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # (Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim)
        return np.array([self.ac_space.sample() for _ in range(num_sequences * horizon)]).reshape((num_sequences, horizon, self.ac_dim)) # HERSHG

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []

        for model in self.dyn_models:
            # (Q2)
            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble
            actions_arr = np.transpose(candidate_action_sequences, (1, 0, 2)) # (horizon, N, self.ac_dim) HERSHG
            obs_arr = np.tile(obs, (self.N, 1))
            states_arr = []
            for i in range(self.horizon):
                obs_arr = model.get_prediction(obs_arr, actions_arr[i], self.data_statistics)
                states_arr.append(obs_arr)
            states_arr = np.array(states_arr)
            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward (See files in envs to see how to call this)
            rewards_arr = [self.env.get_reward(obs_arr, actions_arr[0])[0]]
            for i in range(1, self.horizon-1):
                rewards_arr.append(self.env.get_reward(states_arr[i], actions_arr[i])[0])
            predicted_rewards_per_ens.append(np.array(rewards_arr).sum(axis=0))
        predicted_rewards_per_ens = np.array(predicted_rewards_per_ens)
        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens, N] --> N
        predicted_rewards = predicted_rewards_per_ens.mean(axis=0) # (Q2)

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards) #(Q2)
        best_action_sequence = candidate_action_sequences[best_index] #(Q2)
        action_to_take = best_action_sequence[0] # (Q2)
        return action_to_take[None] # the None is for matching expected dimensions
