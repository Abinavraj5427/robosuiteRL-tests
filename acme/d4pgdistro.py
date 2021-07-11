from acme.agents.tf import d4pg
from acme.tf import networks, utils as tf2_utils
from acme.utils import lp_utils

from acme import wrappers

import gym
import robosuite as suite
from gym import spaces

import numpy as np
import sonnet as snt

import launchpad as lp

"""
Note:
    1. Test with small number of action steps before long training
    2. Make sure robosuite components and environments are up to date
"""

def make_environment(evaluation=False):
    del evaluation

    # Make the robosuite environment
    environment = RSEnv()
    environment = wrappers.GymWrapper(environment) # Converts to dm_env.Environment interface
    environment = wrappers.CanonicalSpecWrapper(environment, clip = True) # Clip action to environment spec
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment

def make_networks(
    action_spec,
    policy_layer_sizes = (256, 256, 256),
    critic_layer_sizes = (512, 512, 256),
    vmin = -150.,
    vmax = 150.,
    num_atoms = 51,
    ):
    # Creates networks used by the agent

    # Get the total dimensions from action spec
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.    
    observation_network = tf2_utils.batch_concat

    # Create the policy network
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])

    # Create the critic network
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        networks.DiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network
    }


def main():
    
    # factories
    environment_factory = lp_utils.partial_kwargs(make_environment)
    network_factory = lp_utils.partial_kwargs(make_networks)

    # program
    program  = d4pg.DistributedD4PG(
        environment_factory = environment_factory,
        network_factory = network_factory,
        num_actors=4,
        max_actor_steps=4000,
    ).build()
    
    lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


class RSEnv(gym.Env):
	"""A robosuite environment for OpenAI Gym"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.env = suite.make(
			env_name = "Lift",
			robots="Sawyer",
			horizon=500,
			has_renderer=False,
			has_offscreen_renderer=False,
			use_camera_obs=False,
			reward_shaping=True
		)
		init_obs = self.env.observation_spec()
		obs_shape = init_obs['robot0_proprio-state'].shape
		obs_low = -np.full(obs_shape, np.inf)
		obs_high = np.full(obs_shape, np.inf)
		lower_bound, upper_bound = self.env.action_spec
		# print(lower_bound.shape, self.env.action_dim)
		self.action_space = spaces.Box(low=lower_bound, high=upper_bound, shape=(self.env.action_dim,), dtype=np.float32)
		self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float32)


	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		return observation['robot0_proprio-state'], reward, done, info

	def reset(self):
		observation = self.env.reset()
		return observation['robot0_proprio-state']

	def render(self, mode='human'):
		self.env.render()

	def close(self):
		self.env.close()





if __name__ == '__main__':
    main()

