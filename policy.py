from flax import linen as nn
import optax
from jax import jit, grad, random, numpy as jnp
from functools import partial
import optax

class Policy():

	def __init__(self):
		self.model = self._architecture()
		self.params = self.model.init(random.PRNGKey(0), jnp.ones((1, 2)))  # Sets up and returns the initial weights and biases

		# Initialize optimizer and state here directly without jit
		gradient_transform = optax.chain(
				optax.clip_by_global_norm(1.0),  # 1.0 is the clipping threshold
				optax.adam(1e-4)  # Learning rate
		)
		self.optimizer = gradient_transform
		self.opt_state = gradient_transform.init(self.params)

	def _architecture(self):
		"""
		contains the NN architecture
		"""
		class PolicyNN(nn.Module):
			# Recommended: 2/3 layers of 128/256
			@nn.compact
			def __call__(self, x):
				hidden_layers = [nn.Dense(features=128) for _ in range(3)]
				for hidden_layer in hidden_layers:
					x = hidden_layer(x)
					x = nn.relu(x)
				x = nn.Dense(features=7)(x)  # 7 q-gates
				return nn.softmax(x)  # Softmax for probability distribution
		return PolicyNN()

	@partial(jit, static_argnums=(0,))
	def compute_gradients(self, actions, states, q_values):

		def loss(params):
			prob_distribution = self.model.apply(params, states)
			chosen_action_probs = prob_distribution[jnp.arange(len(actions)), actions]
			
			# Compute negative log likelihood, weight, average
			log_action_probs = -jnp.log(chosen_action_probs + 1e-8)
			weighted_log_action_probs = log_action_probs * q_values
			reduced_mean_action_probs = jnp.mean(weighted_log_action_probs)
			
			return reduced_mean_action_probs

		return grad(loss)(self.params) # Return the gradient of the loss w.r.t. the NN parameters.

	@partial(jit, static_argnums=(0,))
	def predict(self, input):
		"""
		evaluate the policy
		"""
		return self.model.apply(self.params, input)

	@partial(jit, static_argnums=(0,))
	def _update_params(self, grads):
		updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
		new_params = optax.apply_updates(self.params, updates)
		return new_params, new_opt_state

	@partial(jit, static_argnums=(0,))
	def apply_gradients(self, grads):
		self.params, self.opt_state = self._update_params(grads)