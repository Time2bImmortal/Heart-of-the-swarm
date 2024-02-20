import os  # OS operations
import pandas as pd  # Data Handling
import numpy as np
import tkinter as tk  # GUI (graphic users interface) great for handling paths
from tkinter import filedialog
import re  # String Handling
"""
This module contains two primary classes, each with its distinct responsibility:

1. DataManipulator:
    This class is responsible for managing and processing experiment data files.It provides functionality to select
    individual files or entire folders and categorizes them based on predefined markers. It creates directory structures
    based on slice names and trial combinations, storing organized data accordingly.

2. Calculator:
    This class is designed to compute various behavioral metrics from experimental dataframes. The algorithm are 
    implemented as methods within calculate. 
    Used by DataManipulator.apply_action_to_all_data(Calculator.calculate)

"""


class DataManipulator:
    # Sort the data files and allow the use of specialized classes for getting experiment results
    def __init__(self, delimitation_flags, delimitation_list, trial_dict, trial_dict_2):
        self.delimitation_flags = delimitation_flags
        self.delimitations_list = delimitation_list
        self.trial_dict = trial_dict
        self.trial_dict_2 = trial_dict_2
        self.combinations = {f"{k1}_{k2}": (v1[1], v2[1])  # Create trials combination
                             for k1, v1 in self.trial_dict.items()
                             for k2, v2 in self.trial_dict_2.items()}
        self.result_directories = []
    def choose_files(self):
        root = tk.Tk()
        root.withdraw()
        self.file_paths = filedialog.askopenfilenames(title="Select files")
        root.destroy()
        self._process_paths(self.file_paths)

    def choose_folder(self):
        root = tk.Tk()
        root.withdraw()
        self.folder_path = filedialog.askdirectory()
        root.destroy()
        self._process_paths([self.folder_path])

    def _process_paths(self, paths):
        for path in paths:
            base_name = os.path.basename(path)
            base_dir = os.path.dirname(path)  # Get the directory of the path

            result_dir = os.path.join(base_dir, base_name.replace('marked', 'sliced'))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            self.result_directories.append(result_dir)

            if os.path.isdir(path):  # Separate the keys for trial_dict and trial_dict_2
                for file_name in os.listdir(path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(path, file_name)
                        self.process_file(file_path, result_dir)
            else:
                self.process_file(path, result_dir)

    def apply_action_to_all_data(self, action_method):
        # Walk through all directories and subdirectories and apply an action
        for directory in self.result_directories:
            for root, dirs, _ in os.walk(directory):
                action_method(root)

    def _get_trial_key(self, file_name):
        letter_combination = re.search(r'_([A-Z]_[A-Z])_', file_name).group(1)
        for key, value in self.combinations.items():
            if letter_combination == f'{value[0]}_{value[1]}':
                key1, key2 = key.split('_')  # Separate the keys for trial_dict and trial_dict_2
                return f'{key2.lower()}_for_{key1.lower()}'  # Reversed order
        return None

    def process_file(self, file_path, directory):
        data = pd.read_csv(file_path)
        markers = data['markers']
        flag_markers = [item[0] for item in self.delimitation_flags]
        flag_names = [item[1] for item in self.delimitation_flags]

        original_file_name = os.path.basename(file_path)  # Keeping the original file name
        trial_key = self._get_trial_key(original_file_name)

        for i, marker in enumerate(flag_markers[:-1]):  # skipping the last marker ( end)
            slice_start = markers[markers == marker].index[0]
            slice_end = markers[markers == flag_markers[i + 1]].index[0]

            sliced_data = data.iloc[slice_start:slice_end]
            self.save_slice(sliced_data, flag_names[i], directory, trial_key, original_file_name)

        self.save_slice(data, 'full', directory, trial_key, original_file_name)  # Handle the "full" slice separately

    def save_slice(self, data_slice, slice_name, directory, trial_key, original_file_name):
        # Create directory structure: Folder>sliced_chunk_name>combination>corresponding sliced file
        sub_dir = os.path.join(directory, slice_name, trial_key)
        self._create_subdirectories(sub_dir)

        output_path = os.path.join(sub_dir, original_file_name)

        # Modifications on columns and indices for better use directly on the dataframe
        data_slice = data_slice.drop(data_slice.columns[0], axis=1)
        data_slice.columns = [str(i + 1) for i in range(data_slice.shape[1])]
        data_slice = data_slice.reset_index(drop=True)
        data_slice.index = data_slice.index + 1

        data_slice.to_csv(output_path, index=True)

    @staticmethod
    def _create_subdirectories(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


class Calculator:
    # Calculate features of the subject behavior in each dataframe
    def get_parameters(self, threshold_list, radian_angle, ball_radius, column_mapping):

        if not isinstance(threshold_list, list) or len(threshold_list) != 3:  # Ensure that all parameters are fine
            raise ValueError("Expected a list of three thresholds.")
        for threshold in threshold_list:
            assert isinstance(threshold, (int, float)), "Each threshold should be a number."
        self.axis_threshold = threshold_list

        assert isinstance(radian_angle, (int, float)), "Expected radian_angle to be a number."
        self.radian_angle = radian_angle

        assert isinstance(ball_radius, (int, float)), "Expected ball_radius to be a number."
        self.ball_radius = ball_radius

        assert isinstance(column_mapping, dict), "Expected column_mapping to be a dictionary."
        self.column_mapping = column_mapping

    def pre_process_data(self, df):
        if self.column_mapping:
            df = df.rename(columns=self.column_mapping)

        df.loc[(df['z'] >= 0.0) & (df['z'] <= self.axis_threshold[1]), 'z'] = 0.0
        df.loc[(df['x'] >= 0.0) & (df['x'] <= self.axis_threshold[0]), 'x'] = 0.0
        df.loc[(df['z'] <= 0.0) & (df['z'] >= -self.axis_threshold[2]), 'z'] = 0.0
        df.loc[(df['x'] <= 0.0) & (df['x'] >= -self.axis_threshold[1]), 'x'] = 0.0

        for col in ['x', 'z']:
            df[col] = df[col].rolling(window=10).median()

        return df

    def calculate(self, directory):
        for file in os.listdir(directory):
            if file.endswith('.csv') and file != "computation_summary.csv":
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)
                base_name = os.path.basename(file_path)
                subject_number = base_name[:3].zfill(3)
                combination = os.path.dirname(file_path).split(os.sep)[-1].replace('result_for', '')
                df = self.pre_process_data(df)

                # Calculate metrics
                walking_fraction_value = self.calculate_walking_fraction(df)
                number_of_pauses_value, average_pause_duration_value = self.calculate_pauses_and_average_duration(df)
                lateral_proportion_25_degree = self.calculate_lateral_proportion(df, 90) if walking_fraction_value != 0 else 0
                lateral_proportion_45_degree = self.calculate_lateral_proportion(df, 45) if walking_fraction_value != 0 else 0
                distance_value, position_value = self.calculate_distance_and_position(df) if walking_fraction_value != 0 else (0, 0)
                relative_speed_value = distance_value / walking_fraction_value if walking_fraction_value != 0 else 0

                if isinstance(position_value, (list, tuple)):  # handle the case where position_value is not iterable
                    rounded_pos = tuple(round(x, 2) for x in position_value)
                else:
                    rounded_pos = (round(position_value, 2),)  # I had an issue with 0, so just in case

                result = {  # Store the results in the summary DataFrame
                    'subject': subject_number,
                    'combination': combination,
                    'walking_fraction': round(walking_fraction_value, 2),
                    'number_of_pauses': number_of_pauses_value,
                    'average_pause_duration [s]': round(average_pause_duration_value/25.0, 2),
                    'relative_speed [cm.s-1]': round(relative_speed_value/25.0, 2),
                    'distance_walked [cm]': round(distance_value, 2),
                    'lateral_proportion_25_degree': round(lateral_proportion_25_degree, 2),
                    'lateral_proportion_45_degree': round(lateral_proportion_45_degree, 2),
                    'last_pos': rounded_pos,
                    'chunk_size': len(df)
                    # Others to come...
                }
                directory = os.path.dirname(file_path)
                summary_file_path = os.path.join(directory, 'computation_summary.csv')
                new_result_df = pd.DataFrame([result])
                if not os.path.exists(summary_file_path):
                    new_result_df.to_csv(summary_file_path, index=False)
                else:
                    new_result_df.to_csv(summary_file_path, mode='a', header=False, index=False)  # Append without reading to improve performance

    @staticmethod
    def calculate_walking_fraction(df):
        # Algorithm to extract the proportion of walking activity
        df['is_walking'] = (df['x'] != 0.0) | (df['z'] != 0.0)
        chunk_starts = df.index[df['is_walking'] & ~df['is_walking'].shift(1).fillna(False)].tolist()
        chunk_ends = df.index[df['is_walking'] & ~df['is_walking'].shift(-1).fillna(False)].tolist()

        total_size = sum(end - start + 1 for start, end in zip(chunk_starts, chunk_ends) if (end - start + 1) >= 10)  # Filter chunks by size and compute total size

        return total_size / len(df)  # Return walking fraction

    @staticmethod
    def calculate_pauses_and_average_duration(df):
        # Algorithm to extract the number of pauses and the average time paused
        df['is_paused'] = (df['x'] == 0.0) & (df['z'] == 0.0)

        chunk_starts = df.index[df['is_paused'] & ~df['is_paused'].shift(1).fillna(False)].tolist() # looking for current row is True and the previous False (start pause)
        chunk_ends = df.index[df['is_paused'] & ~df['is_paused'].shift(-1).fillna(False)].tolist() # end pause

        i, valid_chunks = 0, []
        while i < len(chunk_starts): # close small gaps that cant be walking
            start = chunk_starts[i]
            end = chunk_ends[i]
            while i < len(chunk_starts) - 1 and chunk_starts[i + 1] - chunk_ends[i] - 1 < 9:
                i += 1
                end = chunk_ends[i]

            if end - start + 1 >= 10:
                valid_chunks.append((start, end))

            i += 1

        total_pause_duration = sum(e - s + 1 for s, e in valid_chunks)

        number_of_pauses = len(valid_chunks)
        average_duration = total_pause_duration / number_of_pauses if number_of_pauses != 0 else 0

        return number_of_pauses, average_duration  # return the number of pause and the average duration in frames

    def calculate_lateral_proportion(self, df, angle):
        def compute_angles_from_sums(df):
            df['z'] = df['z'].abs()
            # Creating a 'group' column to group every 25 rows together
            df['group'] = df.index // 25
            # Summing the values for every 25 rows
            sum_x_per_group = df.groupby('group')['x'].sum()
            sum_y_per_group = df.groupby('group')['z'].sum()

            angles = np.degrees(np.arctan2(sum_y_per_group.abs(), sum_x_per_group))
            adjusted_angles = np.where(sum_x_per_group < 0, angles, angles)

            return pd.Series(adjusted_angles)

        calculated_angles = compute_angles_from_sums(df)
        walking_df = df[df['is_walking']]
        walking_angles = calculated_angles.loc[walking_df['group'].unique()]

        outside_angle_count = walking_angles[walking_angles > angle].count()

        total_angle_count = len(walking_angles)
        outside_angle_proportion = outside_angle_count / total_angle_count if total_angle_count != 0 else 0

        return outside_angle_proportion

    def calculate_distance_and_position(self, df):
        # Algorithm to calculate the last position and the distance traveled during the experiment
        coords = df[['dis_x', 'dis_y']].to_numpy()
        x_coords, y_coords = self.ball_radius * coords[:, 0], self.ball_radius * coords[:, 1]

        # Adjust the coordinates to the first point (start of the dataframe)
        adjusted_coords = np.column_stack((x_coords, y_coords)) - np.column_stack((x_coords[0], y_coords[0]))

        df['dis_x'], df['dis_y'] = adjusted_coords[:, 0], adjusted_coords[:, 1]

        diffs = df[['dis_x', 'dis_y']].diff().dropna() # Calculate the differences between consecutive rows
        diffs['dis_x'] = np.where(np.abs(diffs['dis_x']) < 0.05, 0.0, diffs['dis_x'])
        diffs['dis_y'] = np.where(np.abs(diffs['dis_y']) < 0.05, 0.0, diffs['dis_y'])

        euclidean_distances = np.sqrt(diffs['dis_x']**2 + diffs['dis_y']**2)  # Calculate the Euclidean distance for each consecutive pair of points

        total_distance_traveled = np.sum(euclidean_distances)  # Calculate the total distance traveled

        final_position = (df['dis_x'].iloc[-1], df['dis_y'].iloc[-1])

        return total_distance_traveled, final_position

