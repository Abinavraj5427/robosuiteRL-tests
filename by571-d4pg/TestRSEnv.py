import gym
import robosuite as suite
from gym import spaces
import numpy as np

class TestRSEnv(gym.Env):
	"""A robosuite environment for OpenAI Gym"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.env = suite.make(
			env_name = "Lift",
			robots="Sawyer",
			horizon=500,
			has_renderer=True,
			has_offscreen_renderer=False,
			use_camera_obs=False,
			reward_shaping=True
		)
		init_obs = self.env.observation_spec()
		#print(init_obs)
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
