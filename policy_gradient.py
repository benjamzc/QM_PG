from jax import tree_map, numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from policy import Policy
from qubit_env import QubitEnv

class Policy_Gradient():

	def __init__(self, epochs):
		self.Env = QubitEnv()
		self.policy = Policy()
		self.epochs = epochs
		self.returns = []

	def sample_action(self, action_probs):
		return np.random.choice(len(action_probs), p=action_probs) 

	def compute_cum_rewards(self, trajectory):
		"""
		Calculate cumulative reward for each step
		"""
		cum_reward = 0
		cum_rewards = []
		for t in reversed(trajectory):
			cum_reward = t[2] + cum_reward
			cum_rewards.insert(0, cum_reward)
		cum_rewards = jnp.array(cum_rewards)
		cum_rewards = (cum_rewards - jnp.mean(cum_rewards)) / (jnp.std(cum_rewards) + 1e-8)

		return cum_rewards

	def update_policy(self, trajectory, cum_rewards):
		# Update policy based on trajectory
		for (state, action, _), cum_reward in zip(trajectory, cum_rewards):
			action = jnp.array(action, dtype=jnp.int32)
			state = jnp.array(state, dtype=jnp.float32)
			grads = self.policy.compute_gradients(action, state) # Compute gradient of the policy wrt the NN parameters

			# Add weight from rewards to gradients
			scaled_gradients = tree_map(lambda x: x * cum_reward, grads)
			self.policy.apply_gradients(scaled_gradients) # Updates the NN parameters in the direction of the gradient
	
	def train_policy(self):
		for epoch in range(self.epochs):
			state = self.Env.reset() # State: random initalize [theta, phi]
			trajectory = []

			# Collect trajectory for one episode
			while True:
				action_probs = self.policy.predict(state) # Probability distribution for 7 gates
				action_probs = np.asarray(action_probs).astype('float64')
				action_probs /= np.sum(action_probs)
				action = self.sample_action(action_probs) # Choose gate with highest probability

				next_state, reward, done = self.Env.step(action) # Perform a single step
				trajectory.append((state, action, reward))
				state = next_state
				if done:
					break

			cum_rewards = self.compute_cum_rewards(trajectory)
			self.update_policy(trajectory, cum_rewards)

			# Total reward for the episode
			episode_return = sum([t[2] for t in trajectory])
			self.returns.append(episode_return)


	def evaluate_policy(self):
		"""
		Evaluate performance of trained policy
		"""
		state = self.Env.reset()
		total_reward = 0
		while True:
			# Normalize probability distribution manually
			action_probs = self.policy.predict(state)
			action_probs = np.asarray(action_probs).astype('float64')
			action_probs /= np.sum(action_probs)
			action = self.sample_action(action_probs)
			next_state, reward, done = self.Env.step(action)
			total_reward += reward
			state = next_state
			if done:
				break
		return total_reward


	def plot_returns(self):
		"""
		Plot the returns as a function of episode number
		"""
		plt.plot(self.returns)
		plt.xlabel('Episode Number')
		plt.ylabel('Total Return')
		plt.title('Performance over iterations')
		plt.show()
