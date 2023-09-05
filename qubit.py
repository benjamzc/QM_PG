import numpy as np
from jax import jit, numpy as jnp
from functools import partial
from qiskit.visualization.bloch import Bloch

class Qubit():
    """
    Custom class which contains the physics
    """
    def __init__(self, T):
        # Initialize the qubit in a random state
        self.init_random_state()

        # Define the different gates
        r_x = jnp.array([[jnp.cos(2 * jnp.pi/T), -1j * jnp.sin(2 * jnp.pi/T)],
                  [-1j * jnp.sin(2 * jnp.pi/T), jnp.cos(2 * jnp.pi/T)]])

        r_y = jnp.array([[jnp.cos(2 * jnp.pi/T), -jnp.sin(2 * jnp.pi/T)],
                  [jnp.sin(2 * jnp.pi/T), jnp.cos(2 * jnp.pi/T)]])

        r_z = jnp.array([[jnp.cos(2 * jnp.pi/T) - (1j * jnp.sin(2 * jnp.pi/T)), 0],
                   [0, jnp.cos(2 * jnp.pi/T) + (1j * jnp.sin(2 * jnp.pi/T))]])

        r_negx = jnp.array([[jnp.cos(2 * jnp.pi/T), 1j * jnp.sin(2 * jnp.pi/T)],
                  [1j * jnp.sin(2 * jnp.pi/T), jnp.cos(2 * jnp.pi/T)]])

        r_negy = jnp.array([[jnp.cos(2 * jnp.pi/T), jnp.sin(2 * jnp.pi/T)],
                  [-jnp.sin(2 * jnp.pi/T), jnp.cos(2 * jnp.pi/T)]])

        r_negz = jnp.array([[jnp.cos(2 * jnp.pi/T) + (1j * jnp.sin(2 * jnp.pi/T)), 0],
                   [0, jnp.cos(2 * jnp.pi/T) - (1j * jnp.sin(2 * jnp.pi/T))]])

        self.gates = jnp.array([r_x, r_negx, r_y, r_negy, r_z, r_negz, jnp.identity(2)])

    def qstate_to_angles(self, qstate):
        # Input qstate : [psi_1, psi_2]
        psi_1 = qstate[0]
        psi_2 = qstate[1]

        theta = 2 * jnp.arccos(jnp.abs(psi_1))
        phi = jnp.angle(psi_2 / jnp.sin(theta / 2))

        return jnp.array([theta, phi])

    @partial(jit, static_argnums=(0,))
    def angles_to_qstate(self, theta, phi):
        # Inputs: theta, phi
        psi_1 = jnp.cos(theta / 2)
        psi_2 = (jnp.cos(phi) + 1j * jnp.sin(phi)) * jnp.sin(theta / 2)

        return jnp.array([psi_1, psi_2])

    def init_random_state(self):
        """
        Initialize the qubit state randomly
        """
        self.phi = 2 * jnp.pi * np.random.uniform(0, 1) # Random phi
        self.theta = jnp.arccos(1 - 2 * np.random.uniform(0, 1)) # Random theta

        self.state = jnp.array([self.theta, self.phi])

    def compute_fidelity(self):
        """
        Fidelity calculated as square inner product of current state and target state
        Range: [0, 1]
        """
        theta = self.state[0]
        inner_prod = jnp.cos(theta/2)
        fidelity = inner_prod ** 2

        return fidelity

    def apply_gate(self, action):
        """
        Gates act on qubits via matrix multiplication
        """
        chosen_gate = self.gates[action]
        qstate = self.angles_to_qstate(self.theta, self.phi)
        new_qstate = jnp.dot(chosen_gate, qstate)

        self.state = self.qstate_to_angles(new_qstate)

    def render_Bloch_repr(self):
        theta = self.state[0]
        phi = self.state[1]

        # Convert theta, phi to Bloch vector
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)

        # Plot the Bloch vector
        bloch = Bloch()
        bloch.add_vectors([x, y, z])
        bloch.add_vectors([0, 0, 1])
        bloch.show()



