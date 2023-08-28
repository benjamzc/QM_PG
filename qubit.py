import numpy as np
from jax import jit, numpy as jnp
from functools import partial
from qiskit.visualization.bloch import Bloch

class Qubit():
    """
    Custom class which contains the physics
    """
    def __init__(self):
        # Initialize the qubit in a random state
        self.init_random_state()

    def init_gates(self, T):
        # Define the different gates
        self.T = T
        self.r_x = [[np.cos(2 * np.pi/self.T), -1j * np.sin(2 * np.pi/self.T)],
                  [-1j * np.sin(2 * np.pi/self.T), np.cos(2 * np.pi/self.T)]]

        self.r_y = [[np.cos(2 * np.pi/self.T), -np.sin(2 * np.pi/self.T)],
                  [np.sin(2 * np.pi/self.T), np.cos(2 * np.pi/self.T)]]

        self.r_z = [[np.exp(-1j * 2 * np.pi/self.T), 0],
                   [0, np.exp(1j * 2 * np.pi/self.T)]]

        return jnp.array([self.r_x, np.negative(self.r_x), self.r_y, np.negative(self.r_y), self.r_z, np.negative(self.r_z), np.identity(2)])

    def qstate_to_angles(self, qstate):
        psi_1 = qstate[0]
        psi_2 = qstate[1]

        theta = 2 * jnp.arccos(np.abs(psi_1))
        phi = jnp.angle(psi_2)

        if phi < 0:
          phi += 2 * jnp.pi

        return jnp.array([theta, phi])

    def angles_to_qstate(self, theta, phi):
        psi_1 = jnp.cos(theta / 2)
        psi_2 = jnp.exp(1j * phi) * jnp.sin(theta / 2)

        return jnp.array([psi_1, psi_2])

    def init_random_state(self):
        """
        Initialize the qubit state randomly
        """
        self.theta = np.random.uniform(0, np.pi)
        self.phi = np.random.uniform(0, 2 * np.pi)
        self.state = jnp.array([self.theta, self.phi])

    @partial(jit, static_argnums=(0,))
    def compute_fidelity(self):
        """
        Fidelity calculated as square inner product of current state and target state
        Range: [0, 1]
        """
        fidelity = 1
        target_state = jnp.array([1, 0])
        if self.state != None:
          qstate = self.angles_to_qstate(self.state[0], self.state[1])
          inner_prod = jnp.abs(jnp.dot(jnp.conj(target_state), qstate))
          fidelity = inner_prod**2

        return fidelity

    def apply_gate(self, action, T):
        """
        Gates act on qubits via matrix multiplication
        """
        gates = self.init_gates(T)
        chosen_gate = gates[action]
        qstate = self.angles_to_qstate(self.theta, self.phi) # Convert to qstate
        new_qstate = jnp.dot(chosen_gate, qstate) # Matrix multiplication: gate, vector
        self.state = self.qstate_to_angles(new_qstate)

    def render_Bloch_repr(self):
        theta = self.state[0]
        phi = self.state[1]
        
        # Convert theta, phi to Bloch sphere coords.
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)
        
        # Plot vectors
        bloch = Bloch()
        bloch.add_vectors([x, y, z]) # Resulting vector
        bloch.add_vectors([0, 0, 1]) # Target vector
        bloch.show()



