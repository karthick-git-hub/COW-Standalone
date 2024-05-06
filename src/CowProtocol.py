import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter

# Quantum states represented by degrees
VACUUM = "decoy"
ALPHA = 0  # Alpha now represents 0 degrees
BETA = 180  # Beta now represents 180 degrees
AMP_DAMP = 45
AMP_DAMP_ERROR_RATE = 0.05
PHASE_ERROR_RATE = 0.05  # Example error rate for phase shifts
QBER_CONSTANT = 11

class QuantumCommunicationSimulator:
    def __init__(self, error_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        self.error_rates = error_rates
        self.reset_simulation()

    def reset_simulation(self):
        self.alice_original_bits_with_timestamps = []
        self.alice_bits_with_timestamps = []
        self.encoded_sequences_with_timestamps = []
        self.bob_data_line_clicks = []
        self.bob_monitoring_line_clicks = []
        self.bob_monitoring_line_clicks_sorted = []
        self.raw_key_alice = []
        self.raw_key_bob = []
        self.valid_timestamps = []
        self.decoy_timestamps = []
        self.encoded_sequences_with_vacuum = []
        self.bit_flip_0_to_180_count = 0
        self.bit_flip_180_to_0_count = 0
        self.total_matches = 0
        self.total_amp_damp_count = 0
        self.total_phase_shift_count = 0
        self.num_to_add = 0
        self.num_segments = 0

    def get_current_timestamp(self):
        return datetime.now()

    def format_timestamp(self, dt):
        return dt.strftime("%d-%b-%Y %H:%M:%S") + ':{:03d}'.format(int(dt.microsecond / 1000))

    # Function to convert timestamp to datetime object
    def parse_timestamp(self, ts):
        return datetime.strptime(ts, "%d-%b-%Y %H:%M:%S:%f")


    def send(self, bits, round_num, photon_count):
        base_timestamp = self.get_current_timestamp()  # Assuming this returns a datetime object
        time_increment = timedelta(microseconds=1000)  # 1 millisecond increment
        for bit in bits:
            # Increment the base timestamp for each bit
            base_timestamp += time_increment
            formatted_timestamp = self.format_timestamp(base_timestamp)
            self.alice_original_bits_with_timestamps.append((bit, formatted_timestamp))
            for _ in range(photon_count):
                self.alice_bits_with_timestamps.append((bit, formatted_timestamp))
        self.encoded_sequences_with_timestamps = [self.encode_bit(bit, timestamp) for bit, timestamp in
                                                  self.alice_bits_with_timestamps]
        print("\n length --- ", len(self.encoded_sequences_with_timestamps),
              "self.encoded_sequences_with_timestamps -- ", self.encoded_sequences_with_timestamps)
        # Call to add_vacuum_sequences_randomly before introducing errors
        self.encoded_sequences_with_vacuum = self.add_vacuum_sequences_randomly()
        print("\n length after vacuum--- ", len(self.encoded_sequences_with_vacuum),
              "self.encoded_sequences_with_vacuum after vacuum -- ", self.encoded_sequences_with_vacuum)
        print("\n `decoy timestamps --- `", self.decoy_timestamps)
        self.adjust_vacuum_timestamps()  # Adjust decoy timestamps after insertion
        print("\n decoy timestamps --- ", self.decoy_timestamps)
        self.encoded_sequences_with_timestamps = [self.introduce_errors(seq_with_ts) for seq_with_ts in
                                                  self.encoded_sequences_with_timestamps]
        print("\n length after errors--- ", len(self.encoded_sequences_with_timestamps),
              "self.encoded_sequences_with_timestamps after errors -- ", self.encoded_sequences_with_timestamps)


    def encode_bit(self, bit, timestamp):
        if bit == 0:
            encoded_sequence = [ALPHA, VACUUM, ALPHA]
        elif bit == 180:
            encoded_sequence = [BETA, VACUUM, BETA]
        return (encoded_sequence, timestamp)


    def add_vacuum_sequences_randomly(self):
        self.num_to_add = 0
        encoded_sequences_with_vacuum = self.encoded_sequences_with_timestamps
        vacuum_sequence = [VACUUM, VACUUM, VACUUM]
        self.num_to_add = round(len(self.encoded_sequences_with_timestamps) * 0.10)
        for _ in range(self.num_to_add):
            insert_index = random.randint(1, len(self.encoded_sequences_with_timestamps) - 2)
            prev_ts = self.parse_timestamp(self.encoded_sequences_with_timestamps[insert_index - 1][1])
            new_ts = prev_ts + timedelta(milliseconds=1)
            formatted_new_ts = self.format_timestamp(new_ts)
            encoded_sequences_with_vacuum.insert(insert_index, (vacuum_sequence, formatted_new_ts))
            self.decoy_timestamps.append(formatted_new_ts)  # Record the decoy timestamp
            for i in range(insert_index + 1, len(encoded_sequences_with_vacuum)):
                sequence, ts = encoded_sequences_with_vacuum[i]
                incremented_ts = self.parse_timestamp(ts) + timedelta(milliseconds=1)
                encoded_sequences_with_vacuum[i] = (sequence, self.format_timestamp(incremented_ts))
            encoded_sequences_with_vacuum.sort(key=lambda x: self.parse_timestamp(x[1]))  # Sort after adjustments
        return encoded_sequences_with_vacuum


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


    def introduce_errors(self, encoded_sequence_with_errors_timestamp):
        encoded_sequence, timestamp = encoded_sequence_with_errors_timestamp
        if random.random() < AMP_DAMP_ERROR_RATE:
            new_sequence = [AMP_DAMP if state != VACUUM else state for state in encoded_sequence]
        else:
            if random.random() < self.error_rate:
                new_sequence = [BETA if state == ALPHA else ALPHA if state == BETA else state for state in encoded_sequence]
            else:
                new_sequence = encoded_sequence
        return (new_sequence, timestamp)


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
            if isinstance(state, int) and state in [0, 180]:
                new_state = state
                # Apply a phase shift based on the PHASE_ERROR_RATE
                if random.random() < PHASE_ERROR_RATE:
                    # Add 90 to the current state to introduce a phase shift
                    new_state = state + 90
                new_sequence.append(new_state)
            else:
                # If not an applicable integer (e.g., 'decoy'), don't modify it
                new_sequence.append(state)
        return (new_sequence, timestamp)


    def receive(self):
        self.bob_data_line_clicks = []
        self.bob_monitoring_line_clicks = []
        self.valid_timestamps = []
        self.beam_splitter(self.encoded_sequences_with_timestamps)
        print("\n self.bob_data_line_clicks.length -- ", len(self.bob_data_line_clicks), "\n self.bob_data_line_clicks -- ", self.bob_data_line_clicks)
        print("\n self.bob_monitoring_line_clicks.length -- ", len(self.bob_monitoring_line_clicks), "\n self.bob_monitoring_line_clicks -- ", self.bob_monitoring_line_clicks)
        # Correctly prepare data for preprocessing
        preprocessed_data_line_clicks = self.preprocess_decode_sequences(self.bob_data_line_clicks, "Data line")
        self.bob_data_line_clicks = preprocessed_data_line_clicks


    def beam_splitter(self, sequence):
        # Ensure the sequence is shuffled randomly
        random.shuffle(sequence)
        # Calculate the split point
        split_point = len(sequence) // 2
        # Split the sequence into two halves
        self.bob_data_line_clicks = self.sort_sequence(sequence[:split_point], "Data Line Clicks")
        self.bob_monitoring_line_clicks = self.sort_sequence(sequence[split_point:], "Monitoring Line Clicks")


    def sort_sequence(self, sequence, line):
        sorted_sequence = sorted(sequence, key=lambda x: self.parse_timestamp(x[1]))
        return sorted_sequence


    def preprocess_decode_sequences(self, preprocessed_sequence, line):
        print("\n Data or Monitoring Line ", line)
        preprocessed_sequence = self.preprocess_sequences(preprocessed_sequence)
        print("\n preprocessed_sequence for " , line , " ---", preprocessed_sequence)
        decoded_sequence = self.bob_decode_mechanism(preprocessed_sequence)
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
            # Now sequences can be counted because they are tuples
            sequence_count = Counter(sequences)
            most_common_sequence = sequence_count.most_common(1)[0][0]
            # Convert back to list if necessary for further processing
            preprocessed_sequences.append((list(most_common_sequence), timestamp))
        return preprocessed_sequences


    def bob_decode_mechanism(self, preprocessed_data_line_clicks):
        decoded_bit = []
        for sequence, timestamp in preprocessed_data_line_clicks:
            # Check for specific sequences [0, 'decoy', 0] and [180, 'decoy', 180]
            if sequence == [0, 'decoy', 0]:
                value = 0
            elif sequence == [180, 'decoy', 180]:
                value = 180
            elif sequence == ['decoy', 'decoy', 'decoy']:
                value = 'decoy'
            # Check if the sequence contains 90, 270, or 45
            elif 90 in sequence:
                value = 90
            elif 270 in sequence:
                value = 270
            elif 45 in sequence:
                value = 45
            else:
                value = None  # Use None or a specific value to indicate an unrecognized sequence

            decoded_bit.append((value, timestamp))
        return decoded_bit


    def ClassicalChannelCommunication(self):
        # Filter out sequences with timestamps that match the decoy_timestamps for Alice
        filtered_sequences_with_timestamps = [seq_with_ts for seq_with_ts in self.encoded_sequences_with_vacuum
                                              if seq_with_ts[1] not in self.decoy_timestamps]
        # Update Alice's raw key based on the filtered sequences
        preprocessed_data_line_clicks = self.preprocess_decode_sequences(filtered_sequences_with_timestamps,
                                                                         "Alice Raw Key")
        self.raw_key_alice = preprocessed_data_line_clicks

        # Filter out Bob's data line clicks that match the decoy_timestamps
        filtered_bob_data_line_clicks = [(bit, timestamp) for bit, timestamp in self.bob_data_line_clicks
                                         if timestamp not in self.decoy_timestamps]

        # Update Bob's raw key with the filtered data line clicks
        self.raw_key_bob = filtered_bob_data_line_clicks

        # Print both Alice's and Bob's raw keys
        print("\nlength --- " , len(self.raw_key_alice), " Alice's raw key (filtered):", self.raw_key_alice)
        print("\nlength --- " , len(self.raw_key_bob), "Bob's raw key (filtered):", self.raw_key_bob)

        # Call compare_results to compare the filtered keys
        return self.compare_results()


    def generate_segments(self, length):
        self.num_segments = 0
        if not isinstance(length, int):
            raise TypeError("Length must be an integer")

        self.num_segments = max(1, round(length * 0.10))  # Calculate 10% of the length, ensuring at least 1 segment
        segments = random.sample(range(length), self.num_segments)  # Generate unique random indices
        return sorted(segments)  # Return the sorted list of segments


    def compare_results(self):
        segments = self.generate_segments(len(self.raw_key_bob))
        print("\nSegments: ", segments)
        self.bit_flip_0_to_180_count = 0
        self.bit_flip_180_to_0_count = 0
        self.total_matches = 0
        self.total_amp_damp_count = 0
        self.total_phase_shift_count = 0  # To track phase shifting errors

        for index in segments:
            alice_bits_by_timestamp = {timestamp: bit for bit, timestamp in self.raw_key_alice}
            bob_bit, bob_timestamp = self.raw_key_bob[index]
            alice_bit = alice_bits_by_timestamp.get(bob_timestamp, None)  # Get Alice's bit by Bob's timestamp
            print(alice_bit)
            print(bob_bit, " --- ", bob_timestamp)
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
            elif bob_bit == AMP_DAMP:
                self.total_amp_damp_count += 1
            # Phase shifting errors - assuming 90 and 270 represent phase shifts
            elif bob_bit in [90, 270]:
                self.total_phase_shift_count += 1

        total_compared = self.total_matches + self.bit_flip_0_to_180_count + self.bit_flip_180_to_0_count + self.total_amp_damp_count + self.total_phase_shift_count
        security_percentage = (self.total_matches / total_compared) * 100 if total_compared > 0 else 0
        print(f"\nTotal compared: {total_compared}, Total matches: {self.total_matches}")
        print(
            f"Total 0 to 180 Bit Flip Errors: {self.bit_flip_0_to_180_count}, Total 180 to 0 Bit Flip Errors: {self.bit_flip_180_to_0_count}")
        print(f"Total AMP_DAMP Errors: {self.total_amp_damp_count}, Total Phase Shift Errors: {self.total_phase_shift_count}")
        print(f"Security percentage: {security_percentage:.2f}%")
        return security_percentage


    def run_simulation(self, rounds_list, photon_counts=[1], bits_count=100):
        self.simulation_results = {}
        self.bits_count = bits_count  # Store the number of bits as an attribute
        for rounds in rounds_list:
            round_results = []  # This will hold the results for each round
            for photon_count in photon_counts:
                photon_count_results = []
                for error_rate in self.error_rates:
                    success_rates = []
                    for round_num in range(1, rounds + 1):
                        self.reset_simulation()  # Reset simulation state for each round
                        self.error_rate = error_rate
                        bits = [random.choice([0, 180]) for _ in range(self.bits_count)]
                        print(f"\nRound {round_num}/{rounds}, Error Rate: {error_rate}, Photon Count: {photon_count}")
                        self.send(bits, round_num, photon_count)
                        self.introduce_phase_errors()
                        self.receive()
                        success_rate = self.ClassicalChannelCommunication()
                        success_rates.append(success_rate)
                        print(f"Round {round_num} Success Rate: {success_rate}%")
                    average_success_rate = np.mean(success_rates) if success_rates else 0
                    photon_count_results.append((error_rate, average_success_rate))
                    print(f"-> Avg Success Rate for Error Rate {error_rate}, Photon Count {photon_count}: {average_success_rate:.2f}%")
                round_results.append((photon_count, photon_count_results))
            self.simulation_results[rounds] = round_results
        self.plot_results(rounds_list)


    def plot_results(self, rounds_list):
        plt.figure(figsize=(10, 6))
        for rounds, round_results in self.simulation_results.items():
            for photon_count, photon_count_results in round_results:
                error_rates, averages = zip(*photon_count_results)
                label = f'Rounds: {rounds}, Photon counts: {photon_count}, Sifting Bit Count: {self.num_segments}'
                plt.plot(error_rates, averages, marker='o', linestyle='-', label=label)

        plt.title(f'Average Success Rate vs. Error Rate\nBits Count: {self.bits_count}')
        plt.xlabel('Error Rate')
        plt.ylabel('Success Rate (%)')
        plt.legend()
        plt.grid(True)
        plt.xticks(sorted(self.error_rates))  # Ensure x-axis ticks cover all error rates
        plt.ylim(0, 100)
        plt.show()


def main():
    start_time = datetime.now()
    print(f"Simulation start time: {start_time}")

    simulator = QuantumCommunicationSimulator()
    rounds_list = [1, 50, 100, 200]  # Specify rounds here if necessary for your logic
    simulator.run_simulation(rounds_list)

    end_time = datetime.now()
    print(f"Simulation end time: {end_time}")
    print(f"Total simulation duration: {end_time - start_time}")

main()
