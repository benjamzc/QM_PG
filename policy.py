from flax import linen as nn
import optax
from jax import jit, grad, random, numpy as jnp
from functools import partial
import optax

class Policy():

	def __init__(self):
		self.model = self._architecture()
		self.params = self.model.init(random.PRNGKey(0), jnp.ones((1, 2)))  # Sets up and returns the initial weights and biases

		# Initialize optimizer
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
			@nn.compact
			def __call__(self, x):
				hidden_layers = [nn.Dense(features=256) for _ in range(5)]
				for hidden_layer in hidden_layers:
					x = hidden_layer(x)
					x = nn.relu(x)
				x = nn.Dense(features=7)(x)  # 7 q-gates
				return nn.softmax(x)  # Softmax for probability distribution
		
		return PolicyNN()

	@partial(jit, static_argnums=(0,))
	def compute_gradients(self, action, input):
		def loss(params):
			prob_vector = self.model.apply(params, input)
			log_prob_action = jnp.log(prob_vector[action] + 1e-8) # select the log probability of the taken action
			return -log_prob_action

		return grad(loss)(self.params) # Return the gradient of the loss w.r.t. the NN parameters.
	
	@partial(jit, static_argnums=(0,))
	def predict(self, input):
		"""
		evaluate the policy
		"""
		return self.model.apply(self.params, input)
	
	@partial(jit, static_argnums=(0,))
	def _update_params(self, scaled_gradients):
		updates, new_opt_state = self.optimizer.update(scaled_gradients, self.opt_state)
		new_params = optax.apply_updates(self.params, updates)
		return new_params, new_opt_state

	def apply_gradients(self, scaled_gradients):
		self.params, self.opt_state = self._update_params(scaled_gradients)