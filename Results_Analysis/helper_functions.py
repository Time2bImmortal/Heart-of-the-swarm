import tkinter as tk
from tkinter import filedialog
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
def print_structure(d, indent=0):
    """Recursively prints the structure of a dictionary."""
    if isinstance(d, dict):
        for key, value in d.items():
            print("  " * indent + f"{key} : {type(value).__name__}")
            print_structure(value, indent + 1)
    elif isinstance(d, (list, tuple, set)):
        if d:  # check if the list is not empty
            print("  " * indent + f"list of {type(d[0]).__name__}")
            print_structure(d[0], indent + 1)
        else:
            print("  " * indent + "empty list")


def create_start_video(video_file, chosen_frame):
    # Show a red start string when chosen frame is selected
    source = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = source.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter('output.avi', fourcc, fps, (1280, 720))

    i = 0
    while True:
        ret, img = source.read()
        if not ret:
            break
        if img.size == 0:
            continue
        if chosen_frame <= i < chosen_frame + 200:
            cv2.putText(img, 'Start', (640, 360), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 2, cv2.LINE_AA)
        output.write(img)
        i += 1

    output.release()
def record_avi_files():
    # Resume in a text all .avi files in a folder
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()

    if folder_path:
        avi_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
        with open(os.path.join(os.path.dirname(folder_path), 'recordings.txt'), 'a') as file:
            for avi_file in avi_files:
                file.write(avi_file + '\n')

        print(f"Recorded {len(avi_files)} .avi files in 'recordings.txt'.")
    else:
        print("No directory selected.")


def record_missing_csv_for_avi():
    # Take two folders, look for csv files that do not have an avi
    root = tk.Tk()
    root.withdraw()

    avi_folder_path = filedialog.askdirectory(title="Select directory containing .avi files")

    csv_folder_path = filedialog.askdirectory(title="Select directory containing .csv files")

    if avi_folder_path and csv_folder_path:
        avi_files_wo_ext = [os.path.splitext(f)[0] for f in os.listdir(avi_folder_path) if f.endswith('.avi')]

        csv_files_wo_ext_and_marked = [os.path.splitext(f)[0].replace('_marked', '') for f in
                                       os.listdir(csv_folder_path) if f.endswith('_marked.csv')]

        missing_csv_files = [avi for avi in avi_files_wo_ext if avi not in csv_files_wo_ext_and_marked]

        with open(os.path.join(os.path.dirname(avi_folder_path), 'recordings.txt'), 'a') as file:
            for avi_file in missing_csv_files:
                file.write(avi_file + '.avi\n')

        print(f"Recorded {len(missing_csv_files)} .avi files without corresponding _marked.csv in 'recordings.txt'.")
    else:
        print("One or both directories were not selected.")


def convert_dat_to_csv():
    # Convert dat files to csv in a directory
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(title="Select directory containing .dat files")

    if not folder_path:
        print("No directory selected.")
        return

    file_paths = [os.path.join(folder_path, filename)
                  for filename in os.listdir(folder_path)
                  if filename.endswith(".dat")]

    if not file_paths:
        print("No .dat files in the selected folder.")
        return

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"Processing filename: {filename}")
        if filename.endswith(".dat"):
            new_filename = filename.split('-')[0] + ".dat"
            new_file_path = os.path.join(os.path.dirname(file_path), new_filename)
            os.rename(file_path, new_file_path)

            df = pd.read_csv(new_file_path, delimiter=',')
            base_filename = os.path.splitext(new_filename)[0]
            df.to_csv(os.path.join(os.path.dirname(file_path), base_filename + '.csv'), index=False)
            print(f"Converted {new_filename} to {base_filename}.csv")


def plot_xyz_from_csv(column_names, axis_rotation_threshold=[0.02, 0.02, 0.02]):
    """
    Open a dialog to select a CSV file and plot the specified columns from the CSV.

    Args:
        column_names (list): A list of three column names to plot.
        axis_rotation_threshold (list, optional): Thresholds for red dashed lines. Defaults to [0.02, 0.02, 0.02].
    """

    root = tk.Tk()
    root.withdraw()  # hide the root window
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])

    if not file_path:
        print("No file selected.")
        return

    df = pd.read_csv(file_path)
    print("Column names in the CSV:", df.columns)

    fig, ax = plt.subplots(3, 1, sharex='all')
    fig.suptitle(f"3-axis motion analysis")

    for j, column in enumerate(column_names):
        ax[j].plot(df[column])
        ax[j].set_ylabel(column)
        ax[j].set_ylim(-0.03, 0.03)

        if j == 0 or j == 2:
            ax[j].axhline(y=axis_rotation_threshold[1], color='crimson', linestyle='--')
            ax[j].axhline(y=-axis_rotation_threshold[0], color='crimson', linestyle='--')
        else:
            ax[j].axhline(y=axis_rotation_threshold[2], color='crimson', linestyle='--')
            ax[j].axhline(y=-axis_rotation_threshold[2], color='crimson', linestyle='--')

    ax[2].set_xlabel('Frames')
    fig.tight_layout()
    plt.show()