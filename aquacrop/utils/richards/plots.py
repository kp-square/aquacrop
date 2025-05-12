import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_crop_simulation_data(df, output_path='crop_simulation_plot.png'):
    """
    Plot crop simulation model data and save to a file.

    Parameters:
    data_path (str): Path to the CSV file containing simulation data
    output_path (str): Path to save the output plot
    """
    # Check if all required columns exist
    required_columns = ['time_step_counter', 'IrrDay', 'Infl', 'Runoff',
                        'DeepPerc', 'Es', 'Tr', 'RootZoneWater']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data")

    # Set up the figure with subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Water balance components subplot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['time_step_counter'], df['Infl'], label='Infiltration', linewidth=2)
    ax1.plot(df['time_step_counter'], df['Runoff'], label='Runoff', linewidth=2)
    ax1.plot(df['time_step_counter'], df['DeepPerc'], label='Deep Percolation', linewidth=2)
    ax1.set_ylabel('Water Amount (mm)', fontsize=12)
    ax1.set_title('Water Balance Components', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Evapotranspiration subplot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['time_step_counter'], df['Es'], label='Soil Evaporation', linewidth=2)
    ax2.plot(df['time_step_counter'], df['Tr'], label='Transpiration', linewidth=2)
    ax2.plot(df['time_step_counter'], df['Es'] + df['Tr'], label='Total ET',
             linewidth=2, linestyle='--', color='green')
    ax2.set_ylabel('Evapotranspiration (mm)', fontsize=12)
    ax2.set_title('Evapotranspiration Components', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Irrigation subplot (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(df['time_step_counter'], df['IrrDay'], color='skyblue', alpha=0.7)
    ax3.set_ylabel('Irrigation (mm)', fontsize=12)
    ax3.set_title('Daily Irrigation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Root zone water subplot (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['time_step_counter'], df['RootZoneWater'], linewidth=2.5, color='darkblue')
    ax4.set_ylabel('Root Zone Water (mm)', fontsize=12)
    ax4.set_title('Root Zone Water Content', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Combined water fluxes (bottom span)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.stackplot(df['time_step_counter'],
                  df['IrrDay'], df['Infl'] - df['IrrDay'], df['Runoff'], df['DeepPerc'],
                  labels=['Irrigation', 'Natural Infiltration', 'Runoff', 'Deep Percolation'],
                  alpha=0.7)
    ax5.plot(df['time_step_counter'], df['Es'] + df['Tr'], label='Total ET',
             linewidth=2.5, color='red')
    ax5.set_xlabel('Time Step', fontsize=12)
    ax5.set_ylabel('Water Flux Components (mm)', fontsize=12)
    ax5.set_title('Combined Water Fluxes', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.suptitle('Crop Simulation Model - Water Balance Components',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


# Example of how to use the function
if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual path to your data file
    plot_crop_simulation_data('your_data.csv', 'crop_simulation_results.png')