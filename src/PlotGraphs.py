import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math


class QuantumCommunicationSimulator:

    def plot_results(self, success_rates_for_distance_alpha, graph_name="5000-rounds-100-bits-success.png"):
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
        plt.title('Key Rate Graph')
        plt.xlabel('Distance between Alice and Bob(km)')
        plt.ylabel('Key rate')
        plt.legend()
        # plt.grid(True)
        plt.ylim(0, 100)
        plt.xticks(range(0, 510, 20))  # Adjust the range as per your data's requirement
        # Save the plot to the file before showing it
        plt.savefig(graph_path)
        # Display the plot
        plt.show()


    def plot_graph_test(self, data_map):
        # Extracting percentages (x-axis) and their corresponding values (y-axis)
        percentages = list(data_map.keys())
        # Applying logarithmic transformation to each value
        values = [- (value * math.log(value)) for value in data_map.values()]

        # Plotting the graph
        plt.figure(figsize=(10, 6))
        plt.plot(percentages, values, marker='o', linestyle='-')
        plt.title('Percentage vs Value')
        plt.xlabel('Percentage')
        plt.ylabel('Value (Log Scale)')
        plt.grid(True)
        plt.show()

def main():
    key_rate_for_distance_alpha = {0.1: [(10.0, 47.64361171029358), (30.0, 48.43504097252936), (50.0, 48.12465367965368), (70.0, 47.691984126984124), (90.0, 47.84333333333333), (110.0, 46.78333333333333), (130.0, 51.45), (150.0, 47.2), (170.0, 48.0), (190.0, 48.1), (210.0, 44.6), (230.0, 39.9), (250.0, 32.6), (270.0, 24.5), (290.0, 16.5), (310.0, 12.8), (330.0, 8.2), (350.0, 4.7), (370.0, 3.5), (390.0, 1.9), (410.0, 2.0), (430.0, 0.9), (450.0, 0.9), (470.0, 0.3), (490.0, 0.0)], 0.15: [(10.0, 48.08521329542244), (30.0, 48.75975857475857), (50.0, 47.56789682539683), (70.0, 47.61999999999999), (90.0, 48.53333333333333), (110.0, 49.45), (130.0, 49.7), (150.0, 41.8), (170.0, 31.1), (190.0, 17.3), (210.0, 11.1), (230.0, 6.1), (250.0, 2.8), (270.0, 0.8), (290.0, 1.0), (310.0, 0.5), (330.0, 0.1), (350.0, 0.1), (370.0, 0.1), (390.0, 0.0), (410.0, 0.0), (430.0, 0.0), (450.0, 0.0), (470.0, 0.0), (490.0, 0.0)], 0.5: [(10.0, 48.94685703185703), (30.0, 45.95), (50.0, 32.1), (70.0, 4.5), (90.0, 0.5), (110.0, 0.0), (130.0, 0.0), (150.0, 0.0), (170.0, 0.0), (190.0, 0.0), (210.0, 0.0), (230.0, 0.0), (250.0, 0.0), (270.0, 0.0), (290.0, 0.0), (310.0, 0.0), (330.0, 0.0), (350.0, 0.0), (370.0, 0.0), (390.0, 0.0), (410.0, 0.0), (430.0, 0.0), (450.0, 0.0), (470.0, 0.0), (490.0, 0.0)]}

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
