import os  # OS operations
import pandas as pd  # Data Handling
import numpy as np
import seaborn as sns  # Plot
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, fligner  # Stats
import scikit_posthocs as sp
import logging  # avoid useless warning of matplotlib by rising the warning level
logging.getLogger("matplotlib").setLevel(logging.ERROR)
from brokenaxes import brokenaxes
"""
This module contains two primary classes, each with its distinct responsibility:

1. Plotter:
    This class is responsible for visualizing the processed experimental data. It provides functionality 
    to summarize data parts, fetch parameters for plotting, and save plots based on trial combinations.
    It generates boxplots for different trial features and saves these visualizations within 
    designated directories. The user can specify custom palettes, features, and trial dictionary 
    mappings to enhance and tailor the visual output.

2. StatisticsAnalyzer:
    This class is designed to compute various statistical metrics and analyses from experimental dataframes.
    It includes utilities for Mann-Whitney U tests, Kruskal-Wallis tests, and others. This class can interpret 
    test results, provide significance indicators based on p-values, and save the analysis results in readable formats.
    Its methods facilitate both pairwise and multi-group statistical comparisons, aiding in the interpretation of
    experimental data.

"""

class Plotter:
    # Plots data files or data results in each data part
    def __init__(self):
        self.data_part_summaries = {}
        self.means = {}

    def get_parameters(self, my_palette, features, trial_dict, trial_dict_2):
        self.my_palette = my_palette
        self.features = features
        self.trial_dict = trial_dict
        self.trial_dict_2 = trial_dict_2

    def apply_action_on_results(self, trial_type_directory):
        # Apply action on result files
        comp_summary_path = os.path.join(trial_type_directory, "computation_summary.csv")

        if not os.path.exists(comp_summary_path):
            return

        data = pd.read_csv(comp_summary_path)
        data_part_directory = os.path.dirname(trial_type_directory)
        data_part_name = os.path.basename(data_part_directory)

        if data_part_name not in self.data_part_summaries:
            self.data_part_summaries[data_part_name] = {"path": data_part_directory, "data": []}

        self.data_part_summaries[data_part_name]["data"].append(data)

    def plot_and_save_data(self, data, save_directory, features):

        data['trial_dict'] = data['combination'].apply(lambda x: x.split('_for_')[1])
        data['trial_dict_2'] = data['combination'].apply(lambda x: x.split('_for_')[0].lower())
        plot_order = [trial.lower() for trial in self.trial_dict]  # Plot the data in the right order

        for feature in features:

            plt.figure(figsize=(21, 6))

            ax = sns.boxplot(data=data, y=feature, x="trial_dict", hue='trial_dict_2', palette=self.my_palette, fliersize=0,
                             order=plot_order,hue_order=['forward', 'backward'])
            sns.swarmplot(data=data, y=feature, x="trial_dict", hue='trial_dict_2', dodge=True, size=3, palette='dark:black',
                          order=plot_order, hue_order=['forward', 'backward'])

            # Add grid and other stylings
            plt.grid(True, which='both', axis='y', color='lightgray', linestyle='-', linewidth=0.35)
            ax.set_axisbelow(True)
            plt.title(f"Feature: {feature}", fontsize=20, fontweight='bold')
            plt.xlabel("Combination", fontsize=20, fontweight='bold')
            plt.ylabel(feature, fontsize=20, fontweight='bold')
            plt.xticks(fontsize=16, fontweight='bold')
            plt.yticks(fontsize=16, fontweight='bold')
            legend_handles = [
                plt.Line2D([0], [0], marker='s', color=self.my_palette[0], label='Forward'),
                plt.Line2D([0], [0], marker='s', color=self.my_palette[1], label='Backward')]
            legend_labels = ['Forward', 'Backward']
            plt.legend(handles=legend_handles, labels=legend_labels, frameon=False)
            plt.savefig(os.path.join(save_directory, f"{feature}.png"))
            plt.close()


    def apply_footprint(self, trial_type_directory):
        """Applies the 2D movement map plot to each dataframe in the directory except for computation_summary.csv.
        It needs to be modified to apply action on data files"""

        valid_files = [filename for filename in os.listdir(trial_type_directory)
                       if filename.endswith(".csv") and filename != "computation_summary.csv"]

        if valid_files:  # If there are valid files, then proceed
            parent_directory = os.path.dirname(trial_type_directory)
            save_directory = os.path.join(parent_directory, 'PLOTS', 'footprints')

            if not os.path.exists(save_directory):  # Ensure the footprints directory exists
                os.makedirs(save_directory)

            for filename in valid_files:
                filepath = os.path.join(trial_type_directory, filename)
                df = pd.read_csv(filepath)
                plot_title = filename[:-4]  # Use the filename (without .csv) as the plot title
                self.plot_2d_movement_map(df, plot_title, save_directory)

    @staticmethod
    def plot_2d_movement_map(dataframe, plot_title, footprint_folder):
        # Create a 2d map of the subject behaviour
        adjusted_coords = dataframe[['14', '15']].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 10))  # Plotting the data
        ax.plot(adjusted_coords[:, 0], adjusted_coords[:, 1], color='white', linewidth=2)
        ax.scatter(adjusted_coords[:, 0], adjusted_coords[:, 1], color='white', marker='o', edgecolors='black')

        ax.set_xlim(min(adjusted_coords[:, 0]) - 1, max(adjusted_coords[:, 0]) + 1)
        ax.set_ylim(min(adjusted_coords[:, 1]) - 1, max(adjusted_coords[:, 1]) + 1)

        ax.set_facecolor('black')
        ax.grid(color='white', linestyle='--', linewidth=0.5)
        fig.canvas.manager.set_window_title(plot_title)

        plt.savefig(os.path.join(footprint_folder, f'{plot_title}.jpg'))  # Save the plot
        plt.close(fig)

    def apply_lateral_mean_deduction(self, trial_type_directory, mode='mean'):
        valid_files = [filename for filename in os.listdir(trial_type_directory)
                       if filename.endswith(".csv") and filename != "computation_summary.csv"]

        if not valid_files:
            print("No valid files found.")
            return

        parent_directory = os.path.dirname(trial_type_directory)
        plot_title = os.path.basename(trial_type_directory)
        direction, trial_type = plot_title.split('_for_')

        if mode == 'mean' or mode == 'both':
            save_directory_mean = os.path.join(parent_directory, 'PLOTS', 'lateral_mean')
            if not os.path.exists(save_directory_mean):
                os.makedirs(save_directory_mean)

            angles_dfs = []
            for filename in valid_files:
                df = self.load_and_process_df(os.path.join(trial_type_directory, filename))
                angles_dfs.append(self.compute_angles_from_sums(df))

            # Calculating the mean of angles at each interval across all dataframes
            mean_angle_df = pd.concat(angles_dfs, axis=1).mean(axis=1)

            if trial_type not in self.means:
                self.means[trial_type] = {}
            self.means[trial_type][direction] = mean_angle_df

            # Check if both directions are present for the trial type
            if len(self.means[trial_type]) == 2:
                # plot both directions together
                self.plot_lateral_mean(trial_type, self.means[trial_type], save_directory_mean, plot_mean=True)

        if mode == 'personal' or mode == 'both':
            save_directory_personal = os.path.join(parent_directory, 'PLOTS', 'lateral_mean_by_subject')
            if not os.path.exists(save_directory_personal):
                os.makedirs(save_directory_personal)

            for filename in valid_files:
                df = self.load_and_process_df(os.path.join(trial_type_directory, filename))
                individual_plot_title = filename[:-4]  # Use filename (without .csv) as plot title
                angles = self.compute_angles_from_sums(df)
                self.plot_lateral_mean(individual_plot_title, angles, save_directory_personal)

    def load_and_process_df(self, filepath):
        df = pd.read_csv(filepath)
        df.loc[(df['7'] >= 0.0) & (df['7'] <= 0.004), '7'] = 0.0
        df.loc[(df['5'] >= 0.0) & (df['5'] <= 0.004), '5'] = 0.0
        df.loc[(df['7'] <= 0.0) & (df['7'] >= -0.004), '7'] = 0.0
        df.loc[(df['5'] <= 0.0) & (df['5'] >= -0.003), '5'] = 0.0

        for col in ['5', '7']:
            df[col] = df[col].rolling(window=5).median()
        df['7'] = df['7'].abs()
        return df

    def compute_angles_from_sums(self, df):
        # Creating a 'group' column to group every 25 rows together
        df['group'] = df.index // 25
        # Summing the values for every 25 rows
        sum_x_per_group = df.groupby('group')['5'].sum()
        sum_y_per_group = df.groupby('group')['7'].sum()

        angles = np.degrees(np.arctan2(sum_y_per_group.abs(), sum_x_per_group))
        adjusted_angles = np.where(sum_x_per_group < 0, angles, angles)

        return pd.Series(adjusted_angles)  # Ensure that the output is a pandas Series

    def plot_lateral_mean(self, plot_title, angles, save_directory, plot_mean=False):
        fig = plt.figure(figsize=(9, 6))  # Store the figure in the 'fig' variable

        # Check if plotting both means together
        if plot_mean:
            direction1, direction2 = angles.keys()
            plt.plot(angles[direction1], marker='o', linestyle='-', color='blue', label=f'{direction1} Direction')
            plt.plot(angles[direction2], marker='o', linestyle='-', color='red', label=f'{direction2} Direction')
        else:
            # Plotting the single direction angles
            plt.plot(angles, marker='o', linestyle='-', color='blue', label='Direction')

        # Setting the y-axis limits to represent the 0 to 180-degree range
        plt.ylim(0, 180)

        # Titles and labels
        plt.title(f"{plot_title} - Mean Direction Progression")
        plt.xlabel('Time')
        plt.ylabel('Direction (Degrees)')

        # Grid for better readability
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Legend
        plt.legend()

        # Layout and saving the figure
        plt.tight_layout()
        filename_suffix = "personal" if "by_subject" in save_directory else "mean"
        plt.savefig(os.path.join(save_directory, f"{plot_title}_lateral_mean_{filename_suffix}.png"))

        plt.close(fig)  # Close the current figure to free up memory

    def finish_and_plot(self):
        for data_part, data_info in self.data_part_summaries.items():
            aggregated_data = pd.concat(data_info["data"], ignore_index=True)  # Concatenate all data for this data
            save_directory = os.path.join(data_info["path"], 'PLOTS')  # Save in the data part all plots in PLOTS
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            self.plot_and_save_data(aggregated_data, save_directory, self.features)


class StatisticsAnalyzer:
    # Calculate and stores the statistics results in each data part
    def __init__(self):
        self.data_part_summaries = {}

    def get_parameters(self, features, trial_dict):
        self.features = features
        self.trial_dict = trial_dict

    @staticmethod
    def get_significance_indicator(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    def number_to_letter(self, num):
        for trial, value in self.trial_dict.items():
            if value[0] == num:
                return value[1]
        return None

    def apply_action(self, trial_type_directory):
        comp_summary_path = os.path.join(trial_type_directory, "computation_summary.csv")

        if not os.path.exists(comp_summary_path):
            return

        data = pd.read_csv(comp_summary_path)
        data_part_directory = os.path.dirname(trial_type_directory)
        data_part_name = os.path.basename(data_part_directory)

        if data_part_name not in self.data_part_summaries:
            self.data_part_summaries[data_part_name] = {
                "path": data_part_directory,
                "data": [],
                "trial_counts": {}
            }

        self.data_part_summaries[data_part_name]["data"].append(data)

        trial_name = os.path.basename(trial_type_directory) # Update trial counts specific to the data part
        if trial_name not in self.data_part_summaries[data_part_name]["trial_counts"]:
            self.data_part_summaries[data_part_name]["trial_counts"][trial_name] = 0
        self.data_part_summaries[data_part_name]["trial_counts"][trial_name] += len(data)

    @staticmethod
    def conduct_mann_whitney_test(data, column, trial_type):
        forward_data = data[data['combination'].str.contains(f"forward_for_{trial_type}")][column]
        backward_data = data[data['combination'].str.contains(f"backward_for_{trial_type}")][column]

        if not len(forward_data) or not len(backward_data):
            return ("Mann-Whitney", None, None)

        u_stat, p_val = mannwhitneyu(forward_data, backward_data)
        return ("Mann-Whitney", p_val, None)

    def conduct_kruskal_wallis_test(self, data, column, trial_types):
        forward_data = [data[data['combination'].str.contains(f"forward_for_{trial_type}")][column].tolist() for
                        trial_type in trial_types]
        backward_data = [data[data['combination'].str.contains(f"backward_for_{trial_type}")][column].tolist() for
                         trial_type in trial_types]

        fligner_forward_pv = fligner(*forward_data).pvalue
        fligner_backward_pv = fligner(*backward_data).pvalue

        kruskal_forward_pv = kruskal(*forward_data).pvalue
        kruskal_backward_pv = kruskal(*backward_data).pvalue

        dunn_forward_df, dunn_backward_df = None, None   # Dunn post-hoc tests (only if Kruskal Wallis is significant)
        if kruskal_forward_pv < 0.05:
            dunn_forward_df = sp.posthoc_dunn(forward_data, p_adjust='bonferroni')
            dunn_forward_df.index = dunn_forward_df.index.map(self.number_to_letter)
            dunn_forward_df.columns = dunn_forward_df.columns.map(self.number_to_letter)

        if kruskal_backward_pv < 0.05:
            dunn_backward_df = sp.posthoc_dunn(backward_data, p_adjust='bonferroni')
            dunn_backward_df.index = dunn_backward_df.index.map(self.number_to_letter)
            dunn_backward_df.columns = dunn_backward_df.columns.map(self.number_to_letter)

        return {
            "fligner_forward_pv": fligner_forward_pv, "fligner_backward_pv": fligner_backward_pv,
            "kruskal_forward_pv": kruskal_forward_pv, "kruskal_backward_pv": kruskal_backward_pv,
            "dunn_forward_df": dunn_forward_df, "dunn_backward_df": dunn_backward_df
        }

    def write_kruskal_wallis_results(self, f, results):

        for feature, result in results.items():
            f.write(f"\n ### {feature} ###\n")
            alpha = 0.05
            if result['fligner_forward_pv'] < alpha or result['fligner_backward_pv'] < alpha:
                f.write(f"    Homogeneity of variances not met:\n")  # Write Fligner results
                f.write(f"        Forward (Fligner test): pv={result['fligner_forward_pv']:.3f}\n")
                f.write(f"        Backward (Fligner test): pv={result['fligner_backward_pv']:.3f}\n\n")

            f.write(
                f"        Forward: pv={result['kruskal_forward_pv']:.3f} {self.get_significance_indicator(result['kruskal_forward_pv'])}\n")
            f.write(
                f"        Backward: pv={result['kruskal_backward_pv']:.3f} {self.get_significance_indicator(result['kruskal_backward_pv'])}\n")

            if result['dunn_forward_df'] is not None:  # Write Dunn posthoc results (if available)
                f.write("\n        Pairwise Comparisons Dunn - Forward Direction:\n")
                f.write(result['dunn_forward_df'].to_string() + "\n")

            if result['dunn_backward_df'] is not None:
                f.write("\n        Pairwise Comparisons - Backward Direction:\n")
                f.write(result['dunn_backward_df'].to_string() + "\n")

    def finish_and_analyze(self):
        for data_part, data_info in self.data_part_summaries.items():
            aggregated_data = pd.concat(data_info["data"], ignore_index=True)
            trial_types = set([name.split("_for_")[1] for name in aggregated_data['combination']])
            results = {}
            kruskal_wallis_results = {}

            for column in self.features:
                results[column] = {}
                for trial_type in trial_types:
                    test_type, p_value, _ = self.conduct_mann_whitney_test(aggregated_data, column, trial_type)
                    significance = self.get_significance_indicator(p_value)
                    results[column][trial_type] = (test_type, p_value, significance)

                kruskal_wallis_results[column] = self.conduct_kruskal_wallis_test(aggregated_data, column, trial_types)

            output_file_path = os.path.join(data_info["path"],
                                            "analysis_results.txt")  # Write the results to a text file
            with open(output_file_path, 'w') as f:
                for trial, count in data_info["trial_counts"].items():
                    f.write(f"{trial}: {count} ~ ")
                f.write("\n\n  ===== MANN-WITHNEY =====\n\n")

                for column, test_info in results.items():  # Write mann-whitney results
                    f.write(f"{column} :\n")
                    for trial_type, details in test_info.items():
                        _, p_val, significance = details
                        if p_val is not None:
                            f.write(f"        {trial_type}: pv={p_val:.3f} {significance}\n")
                    f.write("\n")

                f.write(f"\n  ===== KRUSKAL WALLIS =====\n\n")  # Write Kruskal results
                self.write_kruskal_wallis_results(f, kruskal_wallis_results)






