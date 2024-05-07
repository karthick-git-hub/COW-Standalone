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


# Quantum states represented by degrees
VACUUM = "decoy"
ALPHA = 0  # Alpha now represents 0 degrees
BETA = 180  # Beta now represents 180 degrees
AMP_DAMP_ERROR_RATE = 0.1
PHASE_ERROR_RATE = 0.1  # Example error rate for phase shifts
QBER_CONSTANT = 11

class QuantumCommunicationSimulator:
    def __init__(self):
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
        self.encoded_sequences_with_timestamps = []
        self.encoded_sequences_with_timestamps_error = []
        self.bob_data_line_clicks = []
        self.bob_monitoring_line_clicks = []
        self.bob_monitoring_line_clicks_sorted = []
        self.raw_key_alice = []
        self.raw_key_bob = []
        self.bob_raw_key_calculation = []
        self.valid_timestamps = []
        self.decoy_timestamps = []
        self.encoded_sequences_with_vacuum = []
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


    def send(self, bits, round_num, photon_count, distance):
        base_timestamp = self.get_current_timestamp()  # Assuming this returns a datetime object
        time_increment = timedelta(microseconds=1000)  # 1 millisecond increment
        for bit in bits:
            # Increment the base timestamp for each bit
            base_timestamp += time_increment
            formatted_timestamp = self.format_timestamp(base_timestamp)
            self.alice_original_bits_with_timestamps.append((bit, formatted_timestamp))
            self.alice_bits_with_timestamps.append((bit, formatted_timestamp))
        print("\n length --- ", len(self.alice_bits_with_timestamps),
              "self.alice_bits_with_timestamps -- ", self.alice_bits_with_timestamps)
        self.encoded_sequences_with_timestamps = [self.encode_bit(bit, timestamp) for bit, timestamp in
                                                  self.alice_bits_with_timestamps]
        print("\n length --- ", len(self.encoded_sequences_with_timestamps),
              "self.encoded_sequences_with_timestamps -- ", self.encoded_sequences_with_timestamps)
        # Call to add_vacuum_sequences_randomly before introducing errors
        self.encoded_sequences_with_vacuum = self.add_vacuum_sequences_randomly()
        print("\n length after vacuum--- ", len(self.encoded_sequences_with_vacuum),
              "self.encoded_sequences_with_vacuum after vacuum -- ", self.encoded_sequences_with_vacuum)
        print("\n decoy timestamps --- ", self.decoy_timestamps)
        self.adjust_vacuum_timestamps()  # Adjust decoy timestamps after insertion
        print("\n decoy timestamps --- ", self.decoy_timestamps)
        for bit, timestamp in self.encoded_sequences_with_vacuum:
            for _ in range(photon_count):
                self.encoded_sequences_with_timestamps_error.append((bit, timestamp))
        self.encoded_sequences_with_timestamps_error = [self.introduce_errors(seq_with_ts) for seq_with_ts in
                                                  self.encoded_sequences_with_timestamps_error]
        print("\n length after errors--- ", len(self.encoded_sequences_with_timestamps_error),
              "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps_error)
        self.distanceCalculation(distance)


    def send_via_nodes(self, distance):
        self.encoded_sequences_with_timestamps = [self.introduce_errors(seq_with_ts) for seq_with_ts in
                                                  self.encoded_sequences_with_timestamps]
        print("\n length after errors--- ", len(self.encoded_sequences_with_timestamps),
              "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps)
        self.distanceCalculation(distance)


    def encode_bit(self, bit, timestamp):
        if bit == 0:
            encoded_sequence = [self.qc, VACUUM, self.qc]
        elif bit == 1:
            encoded_sequence = [self.qc_x, VACUUM, self.qc_x]
        return (encoded_sequence, timestamp)


    def add_vacuum_sequences_randomly(self):
        vacuum_sequence = [VACUUM, VACUUM, VACUUM]
        num_vacuums_to_add = round(len(self.encoded_sequences_with_timestamps) * 0.10)
        insert_indices = sorted(
            random.sample(range(1, len(self.encoded_sequences_with_timestamps)), num_vacuums_to_add))

        # Insert vacuum sequences
        for insert_index in reversed(insert_indices):
            prev_ts = self.parse_timestamp(self.encoded_sequences_with_timestamps[insert_index - 1][1])
            new_ts = prev_ts + timedelta(milliseconds=1)
            formatted_new_ts = self.format_timestamp(new_ts)
            self.encoded_sequences_with_timestamps.insert(insert_index, (vacuum_sequence, formatted_new_ts))
            self.decoy_timestamps.append(formatted_new_ts)

        # Ensure every timestamp is at least 1ms greater than the previous
        for i in range(1, len(self.encoded_sequences_with_timestamps)):
            _, current_ts_str = self.encoded_sequences_with_timestamps[i]
            _, prev_ts_str = self.encoded_sequences_with_timestamps[i - 1]
            current_ts = self.parse_timestamp(current_ts_str)
            prev_ts = self.parse_timestamp(prev_ts_str)

            # If current timestamp is not at least 1ms greater, adjust it
            if (current_ts - prev_ts) <= timedelta(milliseconds=0):
                adjusted_ts = prev_ts + timedelta(milliseconds=1)
                self.encoded_sequences_with_timestamps[i] = (
                self.encoded_sequences_with_timestamps[i][0], self.format_timestamp(adjusted_ts))

        # Sort may not be needed if the above logic ensures proper order, but verify to ensure consistency
        # self.encoded_sequences_with_timestamps.sort(key=lambda x: self.parse_timestamp(x[1]))

        return self.encoded_sequences_with_timestamps


    def adjust_vacuum_timestamps(self):
        if not self.encoded_sequences_with_vacuum:
            return  # Exit if the list is empty
        # Initialize or clear the decoy_timestamps list
        self.decoy_timestamps = []
        # Parse the first timestamp
        base_timestamp = self.parse_timestamp(self.encoded_sequences_with_vacuum[0][1])
        # Update all subsequent timestamps by incrementing 1ms each time
        updated_sequences_with_timestamps = []
        for i, (sequence, _) in enumerate(self.encoded_sequences_with_vacuum):
            # Check if the sequence is a decoy sequence
            if sequence == ['decoy', 'decoy', 'decoy']:
                # Save the timestamp for decoy sequences
                self.decoy_timestamps.append(self.format_timestamp(base_timestamp + timedelta(milliseconds=i)))
            # Increment base timestamp by 1ms for each element
            new_timestamp = base_timestamp + timedelta(milliseconds=i)
            formatted_timestamp = self.format_timestamp(new_timestamp)
            updated_sequences_with_timestamps.append((sequence, formatted_timestamp))
        # Update the original list with adjusted timestamps
        self.encoded_sequences_with_vacuum = updated_sequences_with_timestamps


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


    def introduce_phase_errors(self):
        self.encoded_sequences_with_timestamps_error = [self.add_phase_shift_errors(seq_with_ts) for seq_with_ts in
                                                  self.encoded_sequences_with_timestamps_error]
        print("\n length --- ", len(self.encoded_sequences_with_timestamps_error) , "self.encoded_sequences_with_timestamps  after phase Shifting errors -- ", self.encoded_sequences_with_timestamps_error)


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


    def distanceCalculation(self, distance):
        encoded_sequences_with_timestamps = []
        probability = 10 ** (-(self.alpha * distance)/ 10)
        print("\n probability -- ", probability)
        for sequence in self.encoded_sequences_with_timestamps_error:
            if random.random() < probability:
                encoded_sequences_with_timestamps.append(sequence)
        self.encoded_sequences_with_timestamps_error = encoded_sequences_with_timestamps
        print("\n length after distance calculation--- ", len(self.encoded_sequences_with_timestamps_error),
          "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps_error)


    def receive(self):
        self.bob_data_line_clicks = []
        self.bob_monitoring_line_clicks = []
        self.valid_timestamps = []
        self.dm2_clicks = 0
        self.beam_splitter(self.encoded_sequences_with_timestamps_error)
        print("\n self.bob_data_line_clicks.length -- ", len(self.bob_data_line_clicks), "\n self.bob_data_line_clicks -- ", self.bob_data_line_clicks)
        print("\n self.bob_monitoring_line_clicks.length -- ", len(self.bob_monitoring_line_clicks), "\n self.bob_monitoring_line_clicks -- ", self.bob_monitoring_line_clicks)
        # Correctly prepare data for preprocessing
        preprocessed_data_line_clicks = self.preprocess_decode_sequences(self.bob_data_line_clicks, "Data line")
        self.bob_data_line_clicks = preprocessed_data_line_clicks
        dm1, dm2 = self.process_monitoring_line_clicks(self.bob_monitoring_line_clicks)
        print("\n dm1 clicks -- ", dm1, "dm2 clicks -- ", dm2)
        self.dm2_clicks = dm2


    def beam_splitter(self, sequence_with_timestamp):
        for sequence in sequence_with_timestamp:
            if random.random() < 0.7:
                self.bob_data_line_clicks.append(sequence)
            else:
                self.bob_monitoring_line_clicks.append(sequence)


    def sort_sequence(self, sequence, line):
        sorted_sequence = sorted(sequence, key=lambda x: self.parse_timestamp(x[1]))
        return sorted_sequence


    def preprocess_decode_sequences(self, preprocessed_sequence, line):
        print("\n Data or Monitoring Line ", line)
        preprocessed_sequence = self.preprocess_sequences(preprocessed_sequence)
        print("\n length ---", len(preprocessed_sequence),  " preprocessed_sequence for " , line , " ---", preprocessed_sequence)
        decoded_sequence = self.decode_mechanism(preprocessed_sequence)
        print("\n decoded_sequence for " , line , " ---", decoded_sequence)
        return decoded_sequence


    def preprocess_sequences(self, sequences_with_timestamps):
        aggregated_results = {}
        for sequence,  timestamp in sequences_with_timestamps:
            # Convert sequence (list) to tuple for hashing
            sequence_tuple = tuple(sequence)
            if timestamp not in aggregated_results:
                aggregated_results[timestamp] = []
            aggregated_results[timestamp].append((sequence_tuple))
            if timestamp not in self.valid_timestamps:
                self.valid_timestamps.append(timestamp)
        preprocessed_sequences = []
        for timestamp, sequences in aggregated_results.items():
            preprocessed_sequences.append((sequences, timestamp))
        return preprocessed_sequences


    def decode_mechanism(self, preprocessed_data_line_clicks):

        decoded_bit = []
        # Define a function to decode values near 0 or 180
        def decode_value(value):
            if len(value.data) > 0 and value.data[0].operation.name == "x":
                return self.qc_x
            elif len(value.data) > 0 and value.data[0].operation.name == "rz":
                angle = value.data[0].operation.params
                if angle[0] > 4.5:
                    return self.qc_270
                else:
                    return self.qc_90
            else:
                return self.qc
        for sequence, timestamp in preprocessed_data_line_clicks:
            decoded_value = []
            for tuple_elements in sequence:
                decoded_sequence = []
                for value in tuple_elements:
                    if value != 'decoy':
                        decoded_sequence.append(decode_value(value))
                    else:
                        decoded_sequence.append('decoy')
                # Check for specific sequences [0, 'decoy', 0] and [180, 'decoy', 180]
                if decoded_sequence == [self.qc, VACUUM, self.qc]:
                    value = 0
                elif decoded_sequence == [self.qc_x, VACUUM, self.qc_x]:
                    value = 1
                elif decoded_sequence == ['decoy', 'decoy', 'decoy']:
                    value = 'decoy'
                # Check if the sequence contains 90, 270, or 45
                elif decoded_sequence == [self.qc, 'decoy', self.qc_x] or decoded_sequence == [self.qc_x, 'decoy', self.qc]:
                    # Choose randomly between self.qc and self.qc_x
                    value = random.choice([0, 180])
                elif self.qc_90 in decoded_sequence:
                    value = 90
                elif self.qc_270 in decoded_sequence:
                    value = 270
                elif 45 in decoded_sequence:
                    value = 45
                else:
                    value = None  # Use None or a specific value to indicate an unrecognized sequence
                decoded_value.append((value))
            counter = Counter(decoded_value)
            most_common_element, most_common_count = counter.most_common(1)[0]
            decoded_bit.append((most_common_element, timestamp))
        return decoded_bit


    def process_monitoring_line_clicks(self,process_monitoring_line):
        # Initialize counters
        dm1 = 0
        dm2 = 0
        # Loop through each item in the process_monitoring_line list
        for sequence, timestamp in process_monitoring_line:
            # Check if the 1st and 3rd elements of the sequence are the same
            if sequence[0] == sequence[2]:
                dm1 += 1
            else:
                dm2 += 1
        # Return the results
        return dm1, dm2


    def ClassicalChannelCommunication(self, current_round):
        # Filter out Bob's data line clicks that match the decoy_timestamps
        filtered_bob_data_line_clicks = [(bit, timestamp) for bit, timestamp in self.bob_data_line_clicks
                                         if timestamp not in self.decoy_timestamps]
        # Update Bob's raw key with the filtered data line clicks
        self.raw_key_bob = filtered_bob_data_line_clicks
        # Filter out sequences with timestamps that match the decoy_timestamps for Alice
        filtered_sequences_with_timestamps = [seq_with_ts for seq_with_ts in self.encoded_sequences_with_vacuum
                                              if seq_with_ts[1] not in self.decoy_timestamps]
        # Update Alice's raw key based on the filtered sequences
        preprocessed_alice_raw_key = self.preprocess_decode_sequences(filtered_sequences_with_timestamps,
                                                                      "Alice Raw Key")
        # Update Alice's raw key to only include bits with timestamps matching Bob's
        self.raw_key_alice = [(bit, timestamp) for bit, timestamp in preprocessed_alice_raw_key
                              if timestamp in [ts for _, ts in self.raw_key_bob]]

        # Print both Alice's and Bob's raw keys
        print("\nlength --- " , len(self.raw_key_alice), " Alice's raw key (filtered):", self.raw_key_alice)
        print("\nlength --- " , len(self.raw_key_bob), "Bob's raw key (filtered):", self.raw_key_bob)
        self.bob_raw_key_calculation = self.raw_key_bob
        # Call compare_results to compare the filtered keys
        if len(self.raw_key_alice) == 0 or len(self.raw_key_bob) == 0:
            return 0
        else:
            self.sifting_process()
            self.discard_post_sifting()
            return self.parity_check()


    def generate_random_bits(self, length):
        self.random_bits = 0
        if not isinstance(length, int):
            raise TypeError("Length must be an integer")

        self.random_bits = max(1, round(length * 0.10))  # Calculate 10% of the length, ensuring at least 1 segment
        random_bits = random.sample(range(length), self.random_bits)  # Generate unique random indices
        return sorted(random_bits)  # Return the sorted list of random_bits


    def sifting_process(self):
        self.random_bits = self.generate_random_bits(len(self.raw_key_bob))
        print("\nrandom_bits: ", self.random_bits)
        self.bit_flip_0_to_180_count = 0
        self.bit_flip_180_to_0_count = 0
        self.total_matches = 0
        self.total_amp_damp_count = 0
        self.total_phase_shift_count = 0  # To track phase shifting errors
        self.security_percentage = 0

        for index in self.random_bits:
            alice_bits_by_timestamp = {timestamp: bit for bit, timestamp in self.raw_key_alice}
            bob_bit, bob_timestamp = self.raw_key_bob[index]
            alice_bit = alice_bits_by_timestamp.get(bob_timestamp, None)  # Get Alice's bit by Bob's timestamp
            print(f"alice_bits --- {alice_bit}, bob_bit --- {bob_bit}")
            # Matching bits
            if alice_bit == bob_bit:
                self.total_matches += 1
            # Bit flip errors
            elif (alice_bit == ALPHA and bob_bit == BETA) or (alice_bit == BETA and bob_bit == ALPHA):
                if alice_bit == ALPHA:
                    self.bit_flip_0_to_180_count += 1
                else:
                    self.bit_flip_180_to_0_count += 1
            # Amplitude damping errors
            elif bob_bit == 45:
                self.total_amp_damp_count += 1
            # Phase shifting errors - assuming 90 and 270 represent phase shifts
            elif bob_bit in [90, 270]:
                self.total_phase_shift_count += 1

        self.security_percentage  = (self.total_matches / len(self.random_bits)) * 100 if self.total_matches > 0 else 0
        print(f"\nTotal compared: {len(self.random_bits)}, Total matches: {self.total_matches}")
        print(
            f"Total 0 to 180 Bit Flip Errors: {self.bit_flip_0_to_180_count}, Total 180 to 0 Bit Flip Errors: {self.bit_flip_180_to_0_count}")
        print(f"Total AMP_DAMP Errors: {self.total_amp_damp_count}, Total Phase Shift Errors: {self.total_phase_shift_count}")
        print(f"Security percentage: {self.security_percentage :.2f}%")


    def discard_post_sifting(self):
        # Sort the random_bits array in descending order to avoid index shifting issues during removal
        sorted_random_bits = sorted(self.random_bits, reverse=True)
        # For Alice's raw key
        # Assuming self.raw_key_alice is a list of (bit, timestamp) tuples
        for index in sorted_random_bits:
            if index < len(self.raw_key_alice):  # Check to avoid index out of range
                self.raw_key_alice.pop(index)

        # For Bob's raw key
        # Assuming self.raw_key_bob is a list of (bit, timestamp) tuples
        for index in sorted_random_bits:
            if index < len(self.raw_key_bob):  # Check to avoid index out of range
                self.raw_key_bob.pop(index)

        print("\nAfter discarding")
        print("\n length -- ", len(self.raw_key_alice), "self.raw_key_alice", self.raw_key_alice)
        print("\n length -- ", len(self.raw_key_bob), "self.raw_key_bob", self.raw_key_bob)


    def parity_check(self):
        alice_parity_list = []
        bob_parity_list = []
        matched_alice_bits = []
        matched_bob_bits = []
        # Function to calculate parity, appending 0s if necessary
        def calculate_parity(bits):
            # Append 0s to make the length a multiple of 3 for the last group
            while len(bits) % 3 != 0:
                bits.append(0)
            return "even" if sum(bits) % 2 == 0 else "odd"
        # Compute explicit parity for Alice with non-overlapping triples, considering padding
        for i in range(0, len(self.raw_key_alice), 3):
            alice_bits = [self.raw_key_alice[j][0] for j in range(i, min(i + 3, len(self.raw_key_alice)))]
            alice_parity = calculate_parity(alice_bits.copy())  # Use a copy to avoid modifying the original
            alice_parity_list.append(alice_parity)
        # Compute explicit parity for Bob with non-overlapping triples, considering padding
        for i in range(0, len(self.raw_key_bob), 3):
            bob_bits = [self.raw_key_bob[j][0] for j in range(i, min(i + 3, len(self.raw_key_bob)))]
            bob_parity = calculate_parity(bob_bits.copy())  # Use a copy to avoid modifying the original
            bob_parity_list.append(bob_parity)
        print(f"alice_parity_list:", alice_parity_list)
        print("bob_parity_list:", bob_parity_list)
        # Initialize match count
        matches = 0
        # Compare the parity lists directly and count matches
        for i, (a_parity, b_parity) in enumerate(zip(alice_parity_list, bob_parity_list)):
            if a_parity == b_parity:
                matches += 1
                # Calculate start and end indices for actual bits (without padding)
                start_idx = i * 3
                end_idx = start_idx + 3
                # Store the actual bits, excluding the runtime appended 0s
                matched_alice_bits.append(
                    [self.raw_key_alice[j][0] for j in range(start_idx, min(end_idx, len(self.raw_key_alice)))])
                matched_bob_bits.append(
                    [self.raw_key_bob[j][0] for j in range(start_idx, min(end_idx, len(self.raw_key_bob)))])
        print("\n Matches: ", matches)
        # Calculate success rate based on the total pairs compared
        total_pairs = min(len(alice_parity_list), len(bob_parity_list))
        success_rate = (self.count_elements(matched_bob_bits) / len(self.bob_monitoring_line_clicks + self.bob_data_line_clicks))  if len(self.bob_raw_key_calculation) > 0 else 0
        print(f"Explicit parity check success rate: {success_rate}%")
        print("Matched Alice bits:", matched_alice_bits)
        print("Matched Bob bits:", matched_bob_bits)
        return success_rate


    def count_elements(self, nested_list):
        count = 0
        for element in nested_list:
            if isinstance(element, list):
                count += self.count_elements(element)
            else:
                count += 1
        return count


    def shannon_entropy_qber(self, qber):
        if qber <= 0 or qber >= 1:
            raise ValueError("QBER must be between 0 and 1, exclusive.")
        return -qber * math.log2(qber) - (1 - qber) * math.log2(1 - qber)


    def paper(self):
        self.paper_clicks = 0
        print(f"self.random_bits ---  {self.random_bits}")
        print(f"length ---  {len(self.encoded_sequences_with_vacuum)}")
        if(self.random_bits == 0):
             self.paper_clicks = 0
        else:
            self.paper_clicks = math.log(len(self.random_bits) / len(self.encoded_sequences_with_vacuum))
        return self.paper_clicks


    def entropy(self):
        if self.paper_clicks == 0:
            return 0
        else:
            if (self.random_bits == 0):
                return 0
            else:
                return  -(len(self.random_bits) / len(self.encoded_sequences_with_vacuum)) / self.paper_clicks


    def run_simulation(self, distances, rounds=1000, photon_counts=[1], bits_count=500, alphas=[0.10, 0.15, 0.5]):
        self.error_rate = 0.2  # Fixed error rate
        self.simulation_results = {}
        distance_between_alice_and_bob = 0
        nodes = 0
        success_rates_for_distance_alpha = {}
        key_rates_for_distance_alpha = {}
        shannon_entropy_rates_for_distance_alpha = {}
        dm2_clicks_for_distance_alpha = {}
        paper_clicks_for_distance_alpha = {}
        entropy_rates_for_distance_alpha = {}
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

                print(f"\nRunning simulation rounds for distance: {self.distance} km")
                for round_num in range(1, rounds + 1):
                    print(
                        f"\nRound {round_num} of {rounds},  Error Rate: {self.error_rate},  alpha: {alpha}, distance: {distance}")
                    self.reset_simulation()
                    bits = [random.choice([0, 1]) for _ in range(bits_count)]
                    distance_between_alice_and_bob = self.distance
                    self.send(bits, round_num, photon_counts[0], self.distance)
                    self.introduce_phase_errors()
                    for node in range(1, nodes):
                        distance_between_alice_and_bob = self.distance
                        self.send_via_nodes(distance_between_alice_and_bob)
                    self.receive()
                    key_rate = self.ClassicalChannelCommunication(round_num)
                    key_rates.append(key_rate)
                    # success rate - defined using the bits which are same/ bit send to sifting
                    success_rates.append(self.security_percentage)
                    dm2_clicks.append(self.dm2_clicks)
                    paper_clicks.append(self.paper())
                    entropy.append(self.entropy())

                average_success_rate = np.mean(success_rates) if success_rates else 0
                success_rates_for_distance.append((distance_between_alice_and_bob, (average_success_rate)))
                print(f"average_success_rate -- {average_success_rate}")

                average_key_rate = np.mean(key_rates) if key_rates else 0
                print(f"average_key_rate -- {average_key_rate}")
                key_rates_for_distance.append((distance_between_alice_and_bob, (average_key_rate)))

                average_shannon_entropy_rate = self.shannon_entropy_qber((100 - average_success_rate) / 100) if average_success_rate else 0
                shannon_entropy_rates_for_distance.append((distance_between_alice_and_bob, (average_shannon_entropy_rate)))

                average_dm2_clicks = np.mean(dm2_clicks) if dm2_clicks else 0
                dm2_clicks_for_distance.append((distance_between_alice_and_bob,  (average_dm2_clicks)))

                average_paper_clicks = np.mean(paper_clicks) if paper_clicks else 0
                print(f"average_paper_clicks -- {average_paper_clicks}")
                paper_clicks_for_distance.append((distance_between_alice_and_bob, (average_paper_clicks)))

                average_entropy_clicks = np.mean(entropy) if entropy else 0
                print(f"average_entropy_clicks -- {average_entropy_clicks}")
                entropy_rates_for_distance.append((distance_between_alice_and_bob, (average_entropy_clicks)))

            success_rates_for_distance_alpha[alpha] = success_rates_for_distance
            key_rates_for_distance_alpha[alpha] = key_rates_for_distance
            shannon_entropy_rates_for_distance_alpha[alpha] = shannon_entropy_rates_for_distance
            dm2_clicks_for_distance_alpha[alpha] = dm2_clicks_for_distance
            paper_clicks_for_distance_alpha[alpha] = paper_clicks_for_distance
            entropy_rates_for_distance_alpha[alpha] = entropy_rates_for_distance

            print(f"success_rates_for_distance_alpha -- {success_rates_for_distance_alpha}")
            print(f"key_rates_for_distance_alpha -- {key_rates_for_distance_alpha}")
            print(f"shannon_entropy_rates_for_distance_alpha -- {shannon_entropy_rates_for_distance_alpha}")
            print(f"dm2_clicks_for_distance_alpha -- {dm2_clicks_for_distance_alpha}")
            print(f"paper_clicks_for_distance_alpha -- {paper_clicks_for_distance_alpha}")
            print(f"entropy_rates_for_distance_alpha -- {entropy_rates_for_distance_alpha}")

        self.plot_results(key_rates_for_distance_alpha)
        self.storeInFile(success_rates_for_distance_alpha, key_rates_for_distance_alpha,
                         shannon_entropy_rates_for_distance_alpha, dm2_clicks_for_distance_alpha, paper_clicks_for_distance_alpha, entropy_rates_for_distance_alpha)


    def plot_results(self, success_rates_for_distance_alpha, graph_name="100-rounds-100-bits-entropy.png"):
        # Determine the parent directory (assuming this script is in the 'src' folder)
        parent_directory = os.path.join(os.getcwd(), "..")
        # Define the path for the 'graph' folder in the parent directory
        graph_directory = os.path.join(parent_directory, "graph")

        # Check if the 'graph' directory exists, and create it if it doesn't
        if not os.path.exists(graph_directory):
            os.makedirs(graph_directory)
        # Define the full path for the graph image file
        graph_path = os.path.join(graph_directory, graph_name)
        # Plotting the results
        plt.figure(figsize=(10, 6))
        for alpha, success_rates_for_distance in success_rates_for_distance_alpha.items():
            distances, averages = zip(*success_rates_for_distance)
            plt.plot(distances, averages, marker='o', linestyle='-', label=f'Alpha {alpha}')
        plt.title('Average Key Rate vs. Distance ')
        plt.xlabel('Distance (km)')
        plt.ylabel('Key Rate (%)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        plt.xticks(range(0, 501, 10))  # Adjust the range as per your data's requirement
        # Save the plot to the file before showing it
        plt.savefig(graph_path)
        # Display the plot
        plt.show()


    def storeInFile(self, success_rates_for_distance_alpha, key_rates_for_distance_alpha,
                        shannon_entropy_rates_for_distance_alpha, dm2_clicks_for_distance_alpha,
                        paper_clicks_for_distance_alpha, entropy_rates_for_distance_alpha):
        parent_directory = os.path.join(os.getcwd(), "..")
        logs_directory = os.path.join(parent_directory, "logs")
        new_directory = os.path.join(logs_directory, "10000RoundsWith5Nodes")  # Creating a subdirectory 'p2p'
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        success_rates_file = os.path.join(new_directory, "success_rates.txt")
        key_rates_file = os.path.join(new_directory, "key_rates.txt")
        shannon_entropy_rates_file = os.path.join(new_directory, "shannon_entropy_rates.txt")
        dm2_clicks_file = os.path.join(new_directory, "dm2_clicks.txt")
        paper_clicks_file = os.path.join(new_directory, "paper_clicks.txt")
        entropy_rates_file = os.path.join(new_directory, "entropy_rates.txt")

        with open(success_rates_file, "w") as file:
            file.write(f"success_rates_for_distance_alpha -- {success_rates_for_distance_alpha}\n")
        with open(key_rates_file, "w") as file:
            file.write(f"key_rates_for_distance_alpha -- {key_rates_for_distance_alpha}\n")
        with open(shannon_entropy_rates_file, "w") as file:
            file.write(f"shannon_entropy_rates_for_distance_alpha -- {shannon_entropy_rates_for_distance_alpha}\n")
        with open(dm2_clicks_file, "w") as file:
            file.write(f"dm2_clicks_for_distance_alpha -- {dm2_clicks_for_distance_alpha}\n")
        with open(paper_clicks_file, "w") as file:
            file.write(f"paper_clicks_for_distance_alpha -- {paper_clicks_for_distance_alpha}\n")
        with open(entropy_rates_file, "w") as file:
            file.write(f"entropy_rates_for_distance_alpha -- {entropy_rates_for_distance_alpha}\n")


def main():
    start_time = datetime.now()
    print(f"Simulation start time: {start_time}")

    simulator = QuantumCommunicationSimulator()
    start_distance = 10.0  # Starting distance in km
    end_distance = 500.0  # Ending distance in km
    increment = 20.0  # Increment in km
    distances = [start_distance + i * increment for i in range(int((end_distance - start_distance) / increment) + 1)]

    simulator.run_simulation(distances)
    end_time = datetime.now()
    print(f"Simulation end time: {end_time}")
    print(f"Total simulation duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
