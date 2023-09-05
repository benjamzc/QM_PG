from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from policy import Policy
from qubit_env import QubitEnv

class Policy_Gradient():

	def __init__(self, epochs, N):
		self.Env = QubitEnv()
		self.policy = Policy()
		self.epochs = epochs
		self.returns = []
		self.N = N

	def sample_action(self, action_probs):
		return np.random.choice(len(action_probs), p=action_probs)

	def compute_q_values(self, rewards):
		"""
		Calculate cumulative reward for each step
		"""
		cum_reward = 0
		q_values = []
		for t in reversed(rewards):
			cum_reward = t + cum_reward
			q_values.insert(0, cum_reward)

		# Decrease variance trick
		q_values = jnp.array(q_values)
		q_values = q_values - jnp.mean(q_values)

		return q_values

	def single_trajectory(self):
		single_actions, single_states, single_rewards = [], [], []
		state = self.Env.reset()
		# Collect trajectory for one episode
		while True:
			action_probs = self.policy.predict(state) # Probability distribution for 7 gates

			# Manually normalize for precision
			action_probs = np.asarray(action_probs).astype('float64')
			action_probs /= jnp.sum(action_probs)

			action = self.sample_action(action_probs) # Sample action stochastically 
			next_state, reward, done = self.Env.step(action) # Perform a single step
			
			# Update arrays
			single_actions.append(action)
			single_states.append(state)
			single_rewards.append(reward)

			state = next_state
			if done:
				single_q_values = self.compute_q_values(single_rewards)
				break

		return single_actions, single_states, single_q_values, single_rewards

	def sample_trajectories(self, N):
		actions, states, q_values, total_rewards = [], [], [], []
		for samp_epoch in range(N):
			single_actions, single_states, single_q_values, single_rewards = self.single_trajectory()
			actions.extend(single_actions)
			states.extend(single_states)
			q_values.extend(single_q_values)
			total_rewards.extend(single_rewards)

		return jnp.array(actions), jnp.array(states), jnp.array(q_values), jnp.array(total_rewards)

	def update_policy(self, actions, states, q_values):
		grads = self.policy.compute_gradients(actions, states, q_values) # Compute gradient of the policy wrt the NN parameters

		self.policy.apply_gradients(grads) # Updates the NN parameters in the direction of the gradient

	def train_policy(self):
		for epoch in range(self.epochs):
			# For each epoch: obtain NxT matrices for actions, states, q_values
			actions, states, q_values, total_rewards = self.sample_trajectories(self.N)

			# Calc grad of weighted neg log likelihood and update NN parameters
			self.update_policy(actions, states, q_values)

			# Total reward for the episode
			self.returns.append(sum(total_rewards))

	def evaluate_policy(self):
		"""
		Evaluate performance of trained policy
		"""
		state = self.Env.reset()
		total_reward = 0
		while True:
			action_probs = self.policy.predict(state)
			
			# Normalize prob manually
			action_probs = jnp.asarray(action_probs).astype('float64')
			action_probs /= jnp.sum(action_probs)
			
			# Apply action, update to new state
			action = jnp.argmax(action_probs)
			next_state, reward, done = self.Env.step(action)
			total_reward += reward
			state = next_state
			if done:
				break
		self.Env.render()
		# Monitor fidelity of last time step
		print("Last state of evaluation: {}".format(state))
		print("Last evaluation step fidelity: {}".format(reward))
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