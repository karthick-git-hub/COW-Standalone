from collections import Counter
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from qiskit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.instructionset import InstructionSet
from qiskit.converters import circuit_to_instruction
from qiskit.circuit.library import UnitaryGate
from copy import deepcopy


class QuantumCommunicationSimulator:
    def __init__(self):
        self.U_A = None
        self.U_B = None
        self.encoded_sequences_with_timestamps = []
        self.setQuantumBits()
        self.reset_simulation()

    def setQuantumBits(self):
        self.qc = QuantumCircuit(1)
        self.qc_x = QuantumCircuit(1)
        self.qc_x.x(0)
        self.qc_90 = QuantumCircuit(1)
        self.qc_90.rz(math.pi / 2, 0)
        self.qc_270 = QuantumCircuit(1)
        self.qc_270.rz(math.pi * 3 / 2, 0)

    def reset_simulation(self):
        self.alice_original_bits_with_timestamps = []
        self.alice_bits_with_timestamps = []
        self.bob_data_line_clicks = []
        self.bob_monitoring_line_clicks = []
        self.bob_monitoring_line_clicks_sorted = []
        self.raw_key_alice = []
        self.raw_key_bob = []
        self.bob_raw_key_calculation = []
        self.bit_flip_0_to_180_count = 0
        self.bit_flip_180_to_0_count = 0
        self.total_matches = 0
        self.total_amp_damp_count = 0
        self.total_phase_shift_count = 0
        self.num_to_add = 0
        self.random_bits = 0
        self.security_percentage = 0
        self.dm2_clicks = 0
        self.paper_clicks = 0

    def get_current_timestamp(self):
        return datetime.now()

    def format_timestamp(self, dt):
        return dt.strftime("%d-%b-%Y %H:%M:%S") + ':{:03d}'.format(int(dt.microsecond / 1000))

    # Function to convert timestamp to datetime object
    def parse_timestamp(self, ts):
        return datetime.strptime(ts, "%d-%b-%Y %H:%M:%S:%f")

    def generate_unitary(self):
        """Generate a 2x2 unitary diagonal matrix."""
        angle = np.random.rand() * 2 * np.pi
        return np.array([[np.exp(1j * angle), 0],
                         [0, np.exp(-1j * angle)]])

    def is_unitary(self, m):
        """Check if the matrix is unitary."""
        return np.allclose(np.eye(m.shape[0]), m @ m.conj().T)

    def encode_bit(self, bit):
        if bit == 0:
            return self.qc
        elif bit == 1:
            return self.qc_x

    def generate_sequences(self, bit_length):
        alice_bits = np.random.choice([0, 1], bit_length)
        current_time = datetime.now()
        alice_bits_encoded = []
        for bit_value in alice_bits:
            timestamp = current_time.strftime('%d-%b-%Y %H:%M:%S.%f')[:-3]
            encoded_sequence = self.encode_bit(bit_value)
            alice_bits_encoded.append((encoded_sequence, timestamp))
            self.alice_original_bits_with_timestamps.append((bit_value, timestamp))
            current_time += timedelta(milliseconds=1)
        self.encoded_sequences_with_timestamps = alice_bits_encoded


    def send(self, round_num, distance):
        base_timestamp = self.get_current_timestamp()  # Assuming this returns a datetime object
        time_increment = timedelta(microseconds=1000)  # 1 millisecond increment
        new_alice_bits_with_vacuum = []
        for i, (original_bit, timestamp) in enumerate(self.encoded_sequences_with_timestamps):
            # Append the unitary gate to the circuit copy
            circuit_copy = deepcopy(original_bit)
            circuit_copy.append(UnitaryGate(self.U_A, label=f'U_transmission_{i}'), [0])
            new_alice_bits_with_vacuum.append((circuit_copy, timestamp))
        print("\n length --- ", len(new_alice_bits_with_vacuum),
              "self.encoded_sequences_with_timestamps -- ", new_alice_bits_with_vacuum)
        self.encoded_sequences_with_timestamps = [self.introduce_errors(seq_with_ts) for seq_with_ts in
                                                  new_alice_bits_with_vacuum]
        print("\n length after errors--- ", len(self.encoded_sequences_with_timestamps),
              "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps)
        self.distanceCalculation(distance)


    def introduce_errors(self, encoded_sequence_with_timestamp):
        encoded_sequence, timestamp = encoded_sequence_with_timestamp
        new_sequence = []
        for item in encoded_sequence:
            if isinstance(item, QuantumCircuit):
                # Simulate a simple error model: Apply an X gate with a certain probability
                if random.random() < self.error_rate:
                    if len(item.data) > 0 and item.data[0].operation.name == "x":
                        item = self.qc
                    else:
                        item = self.qc_x
                new_sequence.append(item)
            else:
                # For 'decoy' or other non-quantum circuit items, leave as is
                new_sequence.append(item)
        return (new_sequence, timestamp)

    def distanceCalculation(self, distance):
        encoded_sequences_with_timestamps = []
        probability = 10 ** -(self.alpha * distance/ 10)
        print("\n probability -- ", probability)
        for sequence in self.encoded_sequences_with_timestamps:
            if random.random() < probability:
                encoded_sequences_with_timestamps.append(sequence)
        self.encoded_sequences_with_timestamps = encoded_sequences_with_timestamps
        print("\n length after distance calculation--- ", len(self.encoded_sequences_with_timestamps),
          "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps)


    def introduce_phase_errors(self):
        self.encoded_sequences_with_timestamps = [self.add_phase_shift_errors(seq_with_ts) for seq_with_ts in
                                                  self.encoded_sequences_with_timestamps]
        print("\n length --- ", len(self.encoded_sequences_with_timestamps) , "self.encoded_sequences_with_timestamps  after phase Shifting errors -- ", self.encoded_sequences_with_timestamps)


    # Constants for demonstration
    def add_phase_shift_errors(self, encoded_sequence_with_timestamp):
        encoded_sequence, timestamp = encoded_sequence_with_timestamp
        new_sequence = []
        for state in encoded_sequence:
            # Check if the element is an integer and a candidate for a phase shift
            if isinstance(state, QuantumCircuit):
                new_state = state
                # Apply a phase shift based on the PHASE_ERROR_RATE
                if random.random() < PHASE_ERROR_RATE:
                    if len(new_state.data) > 0 and new_state.data[0].operation.name == "x":
                        new_state = self.qc_270
                    else:
                        new_state = self.qc_90
                new_sequence.append(new_state)
            else:
                # If not an applicable integer (e.g., 'decoy'), don't modify it
                new_sequence.append(state)
        return (new_sequence, timestamp)

    def receive(self):
        base_timestamp = self.get_current_timestamp()  # Assuming this returns a datetime object
        time_increment = timedelta(microseconds=1000)  # 1 millisecond increment
        new_alice_bits_with_vacuum = []
        for i, (original_bit, timestamp) in enumerate(self.encoded_sequences_with_timestamps):
            # Append the unitary gate to the circuit copy
            circuit_copy = deepcopy(original_bit[0])
            circuit_copy.append(UnitaryGate(self.U_B, label=f'U_transmission_{i}'), [0])
            new_alice_bits_with_vacuum.append((circuit_copy, timestamp))
        print("\n length --- ", len(new_alice_bits_with_vacuum),
              "self.encoded_sequences_with_timestamps -- ", new_alice_bits_with_vacuum)
        self.encoded_sequences_with_timestamps = [self.introduce_errors(seq_with_ts) for seq_with_ts in
                                                  new_alice_bits_with_vacuum]
        print("\n length after errors--- ", len(self.encoded_sequences_with_timestamps),
              "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps)
        self.distanceCalculation(distance)


    def run_simulation(self, distances, rounds=10, bits_count=10, alphas=[0.10, 0.15, 0.5]):
        self.error_rate = 0.3  # Fixed error rate
        self.simulation_results = {}
        success_rates_for_distance_alpha = {}
        key_rates_for_distance_alpha = {}
        shannon_entropy_rates_for_distance_alpha = {}
        dm2_clicks_for_distance_alpha = {}
        paper_clicks_for_distance_alpha = {}
        entropy_rates_for_distance_alpha = {}
        self.U_A = self.generate_unitary()
        self.U_B = self.generate_unitary()
        print(f"Starting simulation with fixed error rate: {self.error_rate}")
        for alpha in alphas:
            self.alpha = alpha
            success_rates_for_distance = []
            key_rates_for_distance = []
            shannon_entropy_rates_for_distance = []
            dm2_clicks_for_distance = []
            paper_clicks_for_distance = []
            entropy_rates_for_distance = []
            print(f"\nStarting simulation with alpha: {alpha}")
            for distance in distances:
                self.distance = distance
                success_rates = []
                key_rates = []
                dm2_clicks = []
                paper_clicks = []
                entropy = []

                self.generate_sequences(bits_count)

                print(f"\nRunning simulation rounds for distance: {self.distance} km")
                for round_num in range(1, rounds + 1):
                    print(
                        f"\nRound {round_num} of {rounds},  Error Rate: {self.error_rate},  alpha: {alpha}")
                    self.reset_simulation()
                    self.send(round_num, self.distance)
                    self.introduce_phase_errors()
                    self.receive()
                    self.introduce_phase_errors()


def main():
    start_time = datetime.now()
    print(f"Simulation start time: {start_time}")
    simulator = QuantumCommunicationSimulator()
    start_distance = 10.0  # Starting distance in km
    end_distance = 30.0  # Ending distance in km
    increment = 20.0  # Increment in km
    distances = [start_distance + i * increment for i in range(int((end_distance - start_distance) / increment) + 1)]

    simulator.run_simulation(distances)
    end_time = datetime.now()
    print(f"Simulation end time: {end_time}")
    print(f"Total simulation duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
