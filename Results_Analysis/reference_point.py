import os  # File and OS operations
import shutil
import tkinter as tk  # GUI (graphic users interface) great for handling paths
from tkinter import filedialog
import cv2  # Video Processing
import pandas as pd  # Data Handling
import numpy as np
from PIL import Image  # Image Processing
from PIL import ImageChops
from tqdm import tqdm  # Progress Bar
import time  # Handle Time
"""
The ReferenceMarkers class is designed to handle video files, identifying important frames by looking at changes 
in brightness, like when the screen switches between black and white. This helps us determine the start of different 
segments or parts in each video experiment. Alongside this, the class also updates the related CSV files, adding 
markers to highlight these specific segments.
Create an output folder with the same foldername + '_marked' that contains only datafiles delimited, ready for analysis.
A summary of the operations will be saved in a text file within this output directory
"""


class ReferenceMarkers:
    # Some methods are there to allow the to choose files or folder, but It can be used in continue within main.py
    def __init__(self, search_range, contrast_threshold, delimitations_list, chosen_frame_add):
        self.search_range = search_range
        self.contrast_threshold = contrast_threshold
        self.delimitations_list = delimitations_list
        self.chosen_frame_add = chosen_frame_add

    def choose_files(self):
        root = tk.Tk()
        root.withdraw()
        self.file_paths = filedialog.askopenfilenames(title="Select files")
        root.destroy()
        self.get_experiment_name()
        self.create_csv_list()
        self.folder_path= os.path.dirname(self.file_paths[0])
        self.create_output_directory()

    def choose_folder(self):
        root = tk.Tk()
        root.withdraw()
        self.folder_path = filedialog.askdirectory()
        root.destroy()
        self.get_experiment_name()
        self.create_csv_list()
        self.create_output_directory()

    def create_csv_list(self):
        if hasattr(self, 'file_paths'):
            self.csv_files = [file for file in self.file_paths if file.endswith('.csv')]
        if hasattr(self, 'folder_path'):
            self.csv_files = [os.path.join(self.folder_path, filename)
                               for filename in os.listdir(self.folder_path)
                               if filename.endswith('.csv')]

    def get_experiment_name(self):
        self.experiment_name = os.path.basename(self.folder_path)

    def create_output_directory(self):
        for csv_file in self.csv_files:
            video_file = os.path.splitext(csv_file)[0] + '.avi'
            if os.path.exists(video_file):
                self.output_directory = os.path.join(os.path.dirname(self.folder_path), f'{self.experiment_name}_marked')
                os.makedirs(self.output_directory, exist_ok=True)
                return
        print("No .avi file found for the given .csv files.")

    def process_video(self, video_file):
        source = cv2.VideoCapture(video_file)
        Frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        start_range, end_range = self.search_range  # frames range to look for contrast
        end_range = min(end_range, Frames)
        source.set(cv2.CAP_PROP_POS_FRAMES, start_range)
        print('Currently processing: ', video_file, end='\n')
        time.sleep(0.1)
        contrast_list = []
        gray_b = None
        for i in tqdm(range(start_range, end_range)):  # keep track of the progression
            ret, img = source.read()
            if not ret:
                print("Failed to read frame")
                break
            if img.size == 0:
                print("Empty image")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_a = Image.fromarray(gray)
            if gray_b is not None:
                diff = ImageChops.difference(gray_a, gray_b)
                diff = np.mean(np.absolute(np.asarray(diff)))
                contrast_list.append(diff)
            gray_b = gray_a
        return contrast_list

    def get_frame_index(self, contrast_list):
        # look for the contrast above threshold and within the proper range
        contrast_frames = [(int(contrast_list[i]), i + self.search_range[0], (i + self.search_range[0]) / 25) for i in
                           range(len(contrast_list)) if contrast_list[i] > self.contrast_threshold]
        return contrast_frames

    def process_all_videos(self):
        with open(os.path.join(self.output_directory, 'log.txt'), 'w') as log_file:
            log_file.write(str(self.delimitations_list) + '\n')
            for csv_file in self.csv_files:
                video_file = os.path.splitext(csv_file)[0] + '.avi'
                if os.path.exists(video_file):
                    contrast_list = self.process_video(video_file)  # Handle the contrasts found
                    contrast_frames = self.get_frame_index(contrast_list)
                    chosen_frame = contrast_frames[0][1] + self.chosen_frame_add if contrast_frames else None
                    if chosen_frame is not None:
                        # Copy the csv file to the output directory
                        output_csv_file_path = os.path.join(self.output_directory, os.path.basename(csv_file))
                        shutil.copy(csv_file, output_csv_file_path)

                        # Add the markers to the csv file in the output directory
                        data = pd.read_csv(output_csv_file_path)
                        if 'markers' not in data.columns:
                            data['markers'] = None
                        data.at[chosen_frame, 'markers'] = 999
                        for marker, shift in self.delimitations_list:
                            frame_index = chosen_frame + shift
                            if 0 <= frame_index < len(data):
                                data.at[frame_index, 'markers'] = marker
                            else:
                                print(f"Frame index {frame_index} for marker {marker} is out of range. Skipping.")
                                if marker == 111:
                                    data.at[0, 'markers'] = 111
                                    print(f"{marker} has been placed at row 0 in 'markers'")
                        data.to_csv(output_csv_file_path, index=False)

                        log_file.write(f'{csv_file}, {video_file}, chosen frame: {chosen_frame}\n')
                    else:
                        print(f"No suitable frame found for video file: {video_file}")
