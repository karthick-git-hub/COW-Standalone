import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit

class Simulation:
    def __init__(self):
        # Initialize the unitary matrix
        self.U_A = self.generate_unitary()

    def generate_unitary(self):
        """Generate a 2x2 unitary diagonal matrix."""
        angle = np.random.rand() * 2 * np.pi
        return np.array([[np.exp(1j * angle), 0],
                         [0, np.exp(-1j * angle)]])

    def apply_unitary(self, circuit, label_suffix):
        """Applies the unitary gate to the given circuit."""
        circuit.append(UnitaryGate(self.U_A, label=f'U_transmission_{label_suffix}'), [0])

# Example usage
simulation = Simulation()
circuit = QuantumCircuit(1)  # assuming a single-qubit circuit for demonstration
simulation.apply_unitary(circuit, label_suffix=0)

print(circuit)
