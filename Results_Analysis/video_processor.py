import glob  # File and OS operations
import os
import shutil
import tkinter as tk  # GUI (graphic users interface) great for handling paths
from tkinter import filedialog
from threading import Thread, Event  # Concurrency
import subprocess  # Terminal Communication
import cv2 # Video Processing
import pandas as pd  # Data Handling / CheckPoint
import json
import re  # String Handling
import time  # Time
"""
This module contains three primary classes, each with its distinct responsibility:

1. VideoProcessor:
    This class is responsible for managing and converting video files. 
    It provides functionality to choose files or folders and convert them using ffmpeg (installation required: https://ffmpeg.org/download.html).
    It creates a directory with the experiment name and stores the converted video files there.

2. VideoRenamer:
    A utility for comfortably renaming the converted videos, facilitating later sorting.

3. FictracExtractor:
    A utility class that provides tools for managing and processing video data 
    related to the Fictrac software. It allows the user to select video files 
    or folders, process them with Fictrac, manage configurations, and convert 
    Fictrac output data into CSV format. (installation required: https://github.com/rjdmoore/fictrac)

Note: Ensure all required dependencies are installed and the necessary files are present 
in the appropriate directories before using these classes.
"""

class VideoProcessor:
    """ Handle raw video files and convert them to a lighter format. Keep track of the parameters used. Several options
    to allow the user to work comfortably."""

    def __init__(self, experiment_name, convert_parameters, min_max):
        # Initialize the class attributes
        self.experiment_name = experiment_name
        self.convert_parameters = convert_parameters
        self.min_max = min_max

    def process_and_convert(self, experiment_folder_path):
        # Multiple folders processing
        self._create_experiment_folder_full_path(experiment_folder_path)
        self.folders = []
        while True:
            folder = self._choose_single_folder()
            if folder:
                self.folders.append(folder)
                continue_input = input("Press Enter to choose another folder or any key + Enter to stop: ")
                if continue_input.strip() != "":
                    break
            else:
                break

        for folder_path in self.folders:
            for filename in os.listdir(folder_path):
                file_path = os.path.normpath(os.path.join(folder_path, filename))
                if filename.endswith(self.convert_parameters[0]) and self._filter_by_size(file_path):
                    self._convert_file(file_path)

    def _filter_by_size(self, file_path):
        file_size = os.path.getsize(file_path)
        return self.min_max[0] <= file_size <= self.min_max[1]

    @staticmethod
    def _choose_single_folder():
        # open a filedialog to choose a folder
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        root.destroy()
        return folder_path

    def _create_experiment_folder_full_path(self, base_path):
        # Create the experiment folder path using the base_path and the experiment_name
        self.experiment_folder_path = base_path + f'{self.experiment_name}'
        os.makedirs(self.experiment_folder_path, exist_ok=True)

    def choose_files(self):
        root = tk.Tk()
        root.withdraw()
        self.file_paths = filedialog.askopenfilenames(title="Select files")
        self._create_experiment_folder(os.path.dirname(self.file_paths[0]))
        root.destroy()

    def choose_folder(self):
        root = tk.Tk()
        root.withdraw()
        self.folder_path = filedialog.askdirectory()
        self._create_experiment_folder(self.folder_path)
        root.destroy()

    def _create_experiment_folder(self, base_path):
        # Create an experiment folder in the parent directory of the chosen path
        parent_directory = os.path.dirname(base_path)
        self.experiment_folder_path = os.path.join(parent_directory, self.experiment_name)
        os.makedirs(self.experiment_folder_path, exist_ok=True)

    def convert_files(self):
        if hasattr(self, 'folder_path'):
            for filename in os.listdir(self.folder_path):
                if filename.endswith(self.convert_parameters[0]):
                    file_path = os.path.normpath(os.path.join(self.folder_path, filename))
                    self._convert_file(file_path)
        elif hasattr(self, 'file_paths'):
            for file_path in self.file_paths:
                if file_path.endswith(self.convert_parameters[0]):
                    self._convert_file(file_path)
        else:
            print('No files or folders chosen for conversion.')

    def _convert_file(self, file_path):
        # Convert the video file using ffmpeg according to the specified conversion parameters
        filename = os.path.basename(file_path)
        filename_without_extension = os.path.splitext(filename)[0]
        new_filename = filename_without_extension + self.convert_parameters[1]
        new_filepath = os.path.join(self.experiment_folder_path, new_filename)
        subprocess.run(['ffmpeg', '-i', file_path, '-r', str(self.convert_parameters[2]), '-s',
                        str(self.convert_parameters[3]), '-b:v', str(self.convert_parameters[4]), new_filepath])
        print(f'File converted: {new_filename}')

    def save_parameters_to_file(self):
        # Keep track of the parameters used
        param_file_path = os.path.join(self.experiment_folder_path, "parameters.txt")
        with open(param_file_path, 'w') as f:
            f.write(f"Experiment name: {self.experiment_name}\n")
            f.write(f"Conversion parameters: {self.convert_parameters}\n")

    def get_experiment_folder_path(self):
        if hasattr(self, 'experiment_folder_path'):
            return self.experiment_folder_path
        else:
            raise ValueError("Experiment folder has not been created yet.")


class VideoRenamer:
    """Open the videos and communicate with user to rename properly the video files """
    def __init__(self, experiment_folder_path, trial_dict, trial_dict_2, start_num_video, extension):
        self.experiment_folder_path = experiment_folder_path
        self.trial_dict = trial_dict
        self.trial_dict_2 = trial_dict_2
        self.video_number = start_num_video
        self.extension = extension
        self.quarantine_folder = os.path.join(self.experiment_folder_path, "quarantine") # if there is an issue, allow to store the video in a separate place
        self.total_videos = len(glob.glob(os.path.join(self.experiment_folder_path, f'*{self.extension}'))) # just count the total number of videos to rename
        os.makedirs(self.quarantine_folder, exist_ok=True)

    def rename_and_move(self, file_path, subject_number, choice, second_choice):
        # Rename and move the file to its new location
        subject_number = str(subject_number).zfill(3)
        video_number = str(self.video_number).zfill(3)
        new_name = f"{subject_number}_{choice}_{second_choice}_{video_number}{self.extension}"
        new_file_path = os.path.join(self.experiment_folder_path, new_name)
        os.rename(file_path, new_file_path)
        self.video_number += 1
        print(f'Renamed file: {new_file_path}')

    def quarantine(self, file_path):
        # Move a file to the quarantine folder
        shutil.move(file_path, self.quarantine_folder)
        print(f'Moved file to quarantine: {file_path}')

    def open_video_thread(self):
        # Open a video in a thread, allow to look at the video, and renaming it simultaneously

        video_files = glob.glob(os.path.join(self.experiment_folder_path, f'*{self.extension}')) # Get all video files with the specified extension
        for i, file_path in enumerate(video_files, start=1):
            print(f"\nProcessing video {i}/{self.total_videos}")
            print(f"Current video path: {file_path}")
            # create a thread and a way to stop it (stop_event)
            self.stop_event = Event()
            thread = Thread(target=self.open_video, args=(file_path,))
            thread.start()

            while True:
                if not thread.is_alive():
                    time.sleep(1) # Give a little time window to ensure the computer does it properly
                    break
                print("Enter subject number (up to 3 digits): ")
                subject_number = input().strip()
                if not subject_number.isdigit() or len(subject_number) > 3:
                    print("Invalid subject number. Try again.")
                    continue
                # Communicate through the terminal with user
                print("Enter choice for trial: ", end='')
                print(*[f"{value[0]}: {key}" for key, value in self.trial_dict.items()], sep=', ')
                choice = input().strip()

                if choice.isdigit() and any(value[0] == int(choice) for value in self.trial_dict.values()):
                    choice_value = [value[1] for key, value in self.trial_dict.items() if value[0] == int(choice)][0]
                    print("Enter choice for trial 2: ", end='')
                    print(*[f"{value[0]}: {key}" for key, value in self.trial_dict_2.items()], sep=', ')
                    second_choice = input().strip()
                    if second_choice.isdigit() and any(
                            value[0] == int(second_choice) for value in self.trial_dict_2.values()):
                        second_choice_value = \
                            [value[1] for key, value in self.trial_dict_2.items() if value[0] == int(second_choice)][0]
                        if thread.is_alive():
                            self.stop_event.set()
                            time.sleep(1)
                        self.rename_and_move(file_path, subject_number, choice_value, second_choice_value)
                        break
                    else:
                        print('Invalid second choice. Try again.')
                elif choice.lower() == 'q':  # If the user chooses to quarantine the video, move it to the quarantine folder

                    if thread.is_alive():
                        self.stop_event.set()
                        time.sleep(1)
                    self.quarantine(file_path)
                    break
                elif choice.lower() == 'p':  # Allow the user to skip the current video without renaming
                    print('Passing to next video')
                    if thread.is_alive():
                        self.stop_event.set()
                    break
                else:
                    print('Invalid choice. Try again.')

    def open_video(self, file_path):
        # Opens a video using OpenCV
        if os.path.exists(file_path):
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened() and not self.stop_event.is_set():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)  # starts the video at 3000th frame
                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Video", 400, 400)  # Resizes the video window
                cv2.moveWindow("Video", 860, 0)  # Moves the window to position (860, 0)
                while True and not self.stop_event.is_set():  # Loop the video waiting user input
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow("Video", frame)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    else:
                        time.sleep(3)
                        break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"{file_path} not found.")


class FictracExtractor:
    # Use fictrac to extract data from the videos files and convert them to a .csv format
    def __init__(self, fictrac_directory, pattern):
        self.fictrac = fictrac_directory
        self.pattern = re.compile(f'{pattern}')
        self.file_paths = []

    @staticmethod
    def _get_avi_files(directory):
        return [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith('.avi')]

    def choose_files(self):
        root = tk.Tk()
        root.withdraw()
        selected_files = filedialog.askopenfilenames(title="Select files")
        if selected_files:
            directory = os.path.dirname(selected_files[0])
            self.file_paths.extend(self._get_avi_files(directory))
        root.destroy()

    def choose_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.file_paths.extend(self._get_avi_files(folder_path))
        root.destroy()

    @staticmethod
    def _create_config_folder(base_directory):
        config_folder = os.path.join(base_directory, "configuration")
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
        return config_folder

    def fictrac_process(self):
        script_directory = os.path.dirname(os.path.abspath(__file__))  # This gives the directory of the script
        state_file = os.path.join(script_directory, "processed_files.json")
        processed_files = []

        # If the state file exists, load the processed files from it
        if os.path.isfile(state_file):
            with open(state_file, "r") as file:
                processed_files = json.load(file).get("processed_files", [])

        os.chdir(self.fictrac)

        for path in self.file_paths:
            # Update and configure the config.txt for the first video only
            if path not in processed_files:
                self.update_config(path)
                subprocess.run(["cmd", '/c', 'start', '/B', '/wait', 'configgui.exe', 'config.txt'])
                config_copy_name = os.path.basename(path).split('.')[0] + '_config.txt'

                config_folder = self._create_config_folder(os.path.dirname(path))
                shutil.copy('config.txt', os.path.join(config_folder, config_copy_name))

            # Run Fictrac
            subprocess.run(["cmd", '/c', 'start', '/B', '/wait', 'fictrac'])

            # Add path to the processed files list
            processed_files.append(path)
            with open(state_file, "w") as file:
                json.dump({"processed_files": processed_files}, file)

    @staticmethod
    def update_config(path):
        with open("config.txt", "r") as file:
            lines = file.readlines()
            src_fn_line = next(line for line in lines if 'src_fn' in line)
            index = lines.index(src_fn_line)
            lines[index] = src_fn_line.split(":")[0] + f': {path}\n'

        with open('config.txt', 'w') as file:
            file.writelines(lines)

    def convert_dat_to_csv(self):
        # Check if there are paths in self.file_paths and if so, get the directory of the first path
        if not self.file_paths:
            print("No .avi files have been selected.")
            return

        directory = os.path.dirname(self.file_paths[0])
        dat_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith('.dat')]

        if not dat_files:
            print("No .dat files found in the directory.")
            return

        for file_path in dat_files:
            filename = os.path.basename(file_path)
            print(f"Processing filename: {filename}")

            new_filename = filename.split('-')[0] + ".dat"
            new_file_path = os.path.join(directory, new_filename)

            if new_file_path != file_path:  # Rename only if they are different
                os.rename(file_path, new_file_path)

            df = pd.read_csv(new_file_path, delimiter=',')
            base_filename = os.path.splitext(new_filename)[0]
            df.to_csv(os.path.join(directory, base_filename + '.csv'), index=False)

            print(f"Converted {new_filename} to {base_filename}.csv")



