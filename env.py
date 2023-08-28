from gym import spaces
import jax.numpy as jnp
from qubit import Qubit

class QubitEnv():
    def __init__(self, n_time_steps=600):
        # State space. theta: [0, pi], phi: [0, 2*pi]
        self.state_space = spaces.Box(low=jnp.array([0, 0]),
                                      high=jnp.array([jnp.pi, 2 * jnp.pi]),
                                      dtype=jnp.float32)

        self.action_space = spaces.Discrete(7) # Action space: 7 possible gates as operators
        self.q = Qubit() # Create instance of Qubit() class
        self.n_time_steps = n_time_steps
        self.current_step = 0

    def step(self, action):
        self.q.apply_gate(action, self.n_time_steps) # Gate acts on qstate
        reward = self.q.compute_fidelity()
        self.current_step += 1
        done = self.current_step >= self.n_time_steps

        return self.q.state, reward, done

    def reset(self):
        self.current_step = 0
        self.q = Qubit() # Initilizes qubit in random state

        return self.q.state

    def render(self):
        self.q.render_Bloch_repr()