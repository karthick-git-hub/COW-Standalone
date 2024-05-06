import matplotlib.pyplot as plt
import numpy as np  # Import NumPy for numerical operations
from datetime import datetime, timedelta
import os


class QuantumCommunicationSimulator:
    def plot_results(self, success_rates_for_distance_alpha, graph_name="5000-rounds-100-bits-success.png"):
        # Define markers and linestyles to cycle through for each line
        markers = ['o', 'x', '*']
        linestyles = ['-', '--', '-.']

        # Determine the parent directory (assuming this script is in the 'src' folder)
        parent_directory = os.getcwd()  # Adjusted to current working directory
        # Define the path for the 'graph' folder in the parent directory
        graph_directory = os.path.join(parent_directory, "graph")

        # Check if the 'graph' directory exists, and create it if it doesn't
        if not os.path.exists(graph_directory):
            os.makedirs(graph_directory)
        # Define the full path for the graph image file
        graph_path = os.path.join(graph_directory, graph_name)
        # Plotting the results
        plt.figure(figsize=(10, 6))

        # Use itertools.cycle to cycle through markers and linestyles
        from itertools import cycle
        marker_cycle = cycle(markers)
        linestyle_cycle = cycle(linestyles)

        for alpha, success_rates_for_distance in success_rates_for_distance_alpha.items():
            distances, averages = zip(*success_rates_for_distance)
            # Convert tuples to NumPy arrays for element-wise operations
            distances = np.array(distances)
            averages = np.array(averages)
            qber = (100 - averages) / 100  # Calculate QBER rate

            # Get the next marker and linestyle from the cycle
            marker = next(marker_cycle)
            linestyle = next(linestyle_cycle)

            plt.plot(distances, qber, marker=marker, linestyle=linestyle, label=f'Alpha {alpha}')

       # plt.title('QBER Graph')
        plt.xlabel('Distance between Alice and Bob(km)')
        plt.ylabel('Key rate')
        plt.legend()
        # plt.grid(True)
        plt.ylim(0, 1)  # QBER ranges from 0 to 1
        plt.xticks(
            np.arange(0, max(distances) + 10, step=20))  # Adjust the range and step as per your data's requirement
        # Save the plot to the file before showing it
        plt.savefig(graph_path)
        # Display the plot
        plt.show()


def main():
    key_rate_for_distance_alpha = {0.1: [(10.0, 66.68253968253968), (30.0, 67.25333333333333), (50.0, 65.24666666666667), (70.0, 66.08333333333333), (90.0, 67.35), (110.0, 66.2), (130.0, 64.1), (150.0, 65.7), (170.0, 55.9), (190.0, 48.2), (210.0, 37.2), (230.0, 25.4), (250.0, 18.0), (270.0, 13.0), (290.0, 8.5), (310.0, 4.7), (330.0, 3.1), (350.0, 1.8), (370.0, 1.2), (390.0, 1.1), (410.0, 0.5), (430.0, 0.3), (450.0, 0.2), (470.0, 0.1), (490.0, 0.0)], 0.15: [(10.0, 67.18611111111109), (30.0, 65.96), (50.0, 63.933333333333344), (70.0, 63.75), (90.0, 66.3), (110.0, 61.9), (130.0, 45.4), (150.0, 28.9), (170.0, 17.4), (190.0, 8.3), (210.0, 4.0), (230.0, 2.5), (250.0, 0.8), (270.0, 1.0), (290.0, 0.1), (310.0, 0.1), (330.0, 0.0), (350.0, 0.0), (370.0, 0.0), (390.0, 0.1), (410.0, 0.1), (430.0, 0.0), (450.0, 0.0), (470.0, 0.0), (490.0, 0.0)], 0.5: [(10.0, 68.21666666666667), (30.0, 62.7), (50.0, 20.8), (70.0, 1.8), (90.0, 0.6), (110.0, 0.0), (130.0, 0.0), (150.0, 0.0), (170.0, 0.0), (190.0, 0.0), (210.0, 0.0), (230.0, 0.0), (250.0, 0.0), (270.0, 0.0), (290.0, 0.0), (310.0, 0.0), (330.0, 0.0), (350.0, 0.0), (370.0, 0.0), (390.0, 0.0), (410.0, 0.0), (430.0, 0.0), (450.0, 0.0), (470.0, 0.0), (490.0, 0.0)]}
    start_time = datetime.now()
    print(f"Simulation start time: {start_time}")

    simulator = QuantumCommunicationSimulator()
    simulator.plot_results(key_rate_for_distance_alpha)
    # simulator.plot_graph_test(data_map)
    end_time = datetime.now()
    print(f"Simulation end time: {end_time}")
    print(f"Total simulation duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
