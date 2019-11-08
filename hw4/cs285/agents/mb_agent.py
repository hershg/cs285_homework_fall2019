from .base_agent import BaseAgent
from cs285.models.ff_model import FFModel
from cs285.policies.MPC_policy import MPCPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *


class MBAgent(BaseAgent):
    def __init__(self, sess, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped 
        self.sess = sess
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(sess,
                            self.agent_params['ac_dim'],
                            self.agent_params['ob_dim'],
                            self.agent_params['n_layers'],
                            self.agent_params['size'],
                            self.agent_params['learning_rate'],
                            scope='dyn_model_{}'.format(i))
            self.dyn_models.append(model)

        self.actor = MPCPolicy(sess,
                               self.env,
                               ac_dim = self.agent_params['ac_dim'],
                               dyn_models = self.dyn_models,
                               horizon = self.agent_params['mpc_horizon'],
                               N = self.agent_params['mpc_num_action_sequences'],
                               )

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data/self.ensemble_size)

        for i in range(self.ensemble_size):
            
            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful
            l_idx = i * num_data_per_ens
            r_idx = (i+1) * num_data_per_ens
            observations = ob_no[l_idx:r_idx] # (Q1)
            actions = ac_na[l_idx:r_idx] # (Q1)
            next_observations = next_ob_no[l_idx:r_idx] # (Q1)

            # use datapoints to update one of the dyn_models
            model = self.dyn_models[i] # (Q1)
            loss = model.update(observations, actions, next_observations, self.data_statistics)
            losses.append(loss)

        avg_loss = np.mean(losses)
        return avg_loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
                                'obs_std': np.std(self.replay_buffer.obs, axis=0),
                                'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
                                'acs_std': np.std(self.replay_buffer.acs, axis=0),
                                'delta_mean': np.mean(
                                    self.replay_buffer.next_obs - self.replay_buffer.obs,
                                    axis=0),
                                'delta_std': np.std(
                                    self.replay_buffer.next_obs - self.replay_buffer.obs,
                                    axis=0),
                                }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: The size of the batch returned here is sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(batch_size*self.ensemble_size)