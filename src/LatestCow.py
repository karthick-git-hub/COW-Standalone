import numpy as np
import random
import matplotlib.pyplot as plt

# Quantum states represented by degrees
VACUUM = "decoy"
ALPHA = 0  # Alpha now represents 0 degrees
BETA = 180  # Beta now represents 180 degrees
AMP_DAMP = 45
GAMMA = "error"  # New state to represent an error
BIT_FLIP_FOR_ZERO = 180  # Bit flip for 0 degrees to 90 degrees
BIT_FLIP_FOR_PI = 0  # Bit flip for 180 degrees to 270 degrees (180 + 90)

# Function to encode a bit
def encode_bit(bit):
    # Improved encoding logic
    if bit == 0:
        return [ALPHA, VACUUM, ALPHA]  # More distinct representation for 0
    elif bit == 180:
        return [BETA, VACUUM, BETA]  # More distinct representation for 180
    return [VACUUM, VACUUM, VACUUM]

def introduce_errors(encoded_sequences, error_rate=0.05, round_num=1):
    # Apply amplitude damping every 10th round
    if round_num % 10 == 0:
        for i in range(len(encoded_sequences)):
            # Apply amplitude damping logic to a specific sequence
            # For example, you can randomly choose to replace one of the states with 45 degrees
            if random.random() < 0.5:  # Example condition for applying amplitude damping
                encoded_sequences[i] = [state if state != ALPHA else 45 for state in encoded_sequences[i]]

    else:
        # Regular error introduction logic
        for i in range(len(encoded_sequences)):
            if random.random() < error_rate:  # Error rate determines the likelihood of an error
                # Randomly introducing either a general error or a bit flip error
                if random.random() < 0.3:  # 50% chance for each type of error
                    # General error
                    encoded_sequences[i] = [GAMMA, GAMMA, GAMMA]
                else:
                    # Bit flip error
                    encoded_sequences[i] = [state + BIT_FLIP_FOR_ZERO if state == ALPHA else
                                            state + BIT_FLIP_FOR_PI if state == BETA else
                                            state for state in encoded_sequences[i]]
    return encoded_sequences


def beam_splitter(sequence):
    data_line = []
    monitoring_line = []
    for state in sequence:
        if random.random() < 0.5:  # 50% chance for data line
            data_line.append(state)
        else:  # 50% chance for monitoring line
            monitoring_line.append(state)
    return data_line, monitoring_line


def data_line_decode(sequence):
    # Improved decoding logic
    alpha_count = sequence.count(ALPHA)
    beta_count = sequence.count(BETA)
    if alpha_count > beta_count:
        return 0
    elif beta_count > alpha_count:
        return 180
    elif AMP_DAMP in sequence:
        return AMP_DAMP
    return None


# Function to simulate monitoring line coherence check
def monitoring_line_check(sequence):
    # Coherence check between consecutive non-vacuum and non-error states
    valid_states = [state for state in sequence if state != GAMMA]

    # Check for error state
    if GAMMA in sequence:
        return "ERROR"  # or False, depending on how you want to represent it

    return len(valid_states) >= 1  # True if 0 or 1 valid states, False otherwise

def compare_Alice_Bob_Results(alice_bits, bob_data_line_bits):
    comparison_results = []

    for alice_bit, bob_bit in zip(alice_bits, bob_data_line_bits):
        if alice_bit is not None and bob_bit is not None:
            if alice_bit == bob_bit:
                comparison_results.append(True)
            elif bob_bit == 45:
                comparison_results.append("AMP_DAMP")
            elif (alice_bit == 0 and bob_bit == 180) or (alice_bit == 180 and bob_bit == 0):
                comparison_results.append("Bit Flip")
            else:
                comparison_results.append(False)
        else:
            comparison_results.append('Skipped')  # Skip if either value is None

    return comparison_results


# Function to run the simulation for a given number of rounds
def run_simulation(rounds):
    all_round_percentages = []
    for round_num in range(rounds):
        # Simulate Alice generating bits, encoding, introducing errors, Bob decoding, etc.
        alice_bits = [random.choice([0, 180]) for _ in range(20)]  # Example Alice's bits
        print("Bits Alice sent:", alice_bits)
        encoded_sequences = [encode_bit(bit) for bit in alice_bits]
        print("After encoding:", encoded_sequences)
        encoded_sequences = introduce_errors(encoded_sequences, error_rate=0.05, round_num=round_num)
        print("After encoding and adding errors:", encoded_sequences)
        bob_data_line_bits = []
        monitoring_checks = []

        for seq in encoded_sequences:
            data_line, monitoring_line = beam_splitter(seq)
            bob_data_line_bits.append(data_line_decode(data_line))
            monitoring_checks.append(monitoring_line_check(monitoring_line))

        print("DataLine Output:", bob_data_line_bits)
        print("Monitoring Line Output:", monitoring_checks)

        comparison_results = compare_Alice_Bob_Results(alice_bits, bob_data_line_bits)

        # Calculate the filtered bits, errors, and bit flips
        filtered_bob_bits = [bit for bit in bob_data_line_bits if bit is not None]
        error_or_false_count = sum(check == "ERROR" or check == False for check in monitoring_checks)
        bit_flip_count = sum(result == "Bit Flip" for result in comparison_results)
        amplifier_damping_count = sum(result == "AMP_DAMP" for result in comparison_results)
        # Calculate the success percentage for the current round
        success_percentage = ((len(filtered_bob_bits) - (error_or_false_count + bit_flip_count + amplifier_damping_count)) / len(
            alice_bits)) * 100
        all_round_percentages.append(success_percentage)

        # Printing counts
        print("Total number of bits Alice sent:", len(alice_bits))
        print("Total number of bits Bob received in data line:", len(filtered_bob_bits))
        print("Total number of errored bits Bob received in monitoring line:", error_or_false_count)
        print("Comparison Result:", comparison_results)
        print("Errors:")
        print("Number of bit flips after Bob sends the bits in dataline to Alice for cross-checking:", bit_flip_count)
        print("\nNumber of amplitude damping bits  found in Bob's detection lines:", amplifier_damping_count)

        print(all_round_percentages)
        print("========================================")

    return all_round_percentages  # Return the list of percentages for each round


# Function to plot the results
def plot_results(total_rounds_sets, all_percentages):
    plt.figure(figsize=(12, 8))
    # Plot the results for each set of rounds
    for i, rounds in enumerate(total_rounds_sets):
        round_numbers = np.arange(1, rounds + 1)  # X-axis: Round numbers
        plt.plot(round_numbers, all_percentages[i], label=f'{rounds} Rounds', linestyle='-')

    plt.xlabel('Round Number')
    plt.ylabel('Success Percentage')
    plt.title('Success Percentage for Each Round in Different Sets')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to run the simulation and plot the results
def simulate_and_plot():
    total_rounds_sets = [5]  # Sets of rounds for the simulation
    all_percentages = [run_simulation(rounds) for rounds in total_rounds_sets]  # Run the simulations
    # plot_results(total_rounds_sets, all_percentages)  # Plot the results

simulate_and_plot()  # Call the main function