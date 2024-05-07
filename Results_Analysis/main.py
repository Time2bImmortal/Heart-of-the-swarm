import video_processor as vp
import reference_point as rp
import features_extraction as fe
import plots_and_stats as ps
from extraction_parameters import P
import tkinter
from tkinter import filedialog
"""
Videos to results processing

Modules:
-------
1. video_processor (vp) - Converts video files between formats, mainly from .mp4 to .avi.
2. reference_point (rp) - Utilizes reference markers as the starting point to process data files. 
3. features_extraction (fe) - Facilitates the extraction of essential features from data files.
4. plots_and_stats (ps) - Enables the creation of plots and the computation of statistical analyses.
5. extraction_parameters (P) - Contains various parameters required by the above modules.

Usage:
-----
The typical flow involves:
1. Converting video files.
2. Renaming videos based on user input.
3. Extracting data using Fictrac.
4. Processing data using reference markers.
5. Manipulating data to organize and extract desired features.
6. Plotting the extracted features.
7. Performing statistical analyses on the features.

Note: Fictrac is complex to install, so this script works post features extraction.
"""


if __name__ == "__main__":

    # Upload parameters
    video_params, ref_params = P["video_processor_params"], P["reference_point_params"]
    res_params, plots_params = P["results_extraction"], P["plots_and_stats"]

    """Irrelevant, convert videos"""
    # video_processor = vp.VideoProcessor(video_params['experiment_name'], video_params['convert_params'],
    #                                     video_params['expected_size_file_bits'])
    # # video_processor.process_and_convert("C:\\Users\\yfant\\OneDrive\\Desktop\\special_peer_recognition")
    # video_processor.choose_folder()
    # video_processor.convert_files()
    # video_processor.save_parameters_to_file()
    # folder_name = video_processor.get_experiment_folder_path()
    # folder_name = filedialog.askdirectory()

    """Irrelevant, rename videos"""
    # video_renamer = vp.VideoRenamer(folder_name, video_params["trial_dict"], video_params["trial_dict_2"],
    #                                 video_params["start_num_video"], video_params['convert_params'][1])
    # video_renamer.open_video_thread()

    """Irrelevant, use fictrac"""
    # fictracer = vp.FictracExtractor(video_params['fictrac_directory'], video_params['formatted_name'])
    # fictracer.choose_folder()
    # fictracer.fictrac_process()
    # fictracer.convert_dat_to_csv()

    """Irrelevant, marked the files, test it on pre-marked folder"""
    # marker = rp.ReferenceMarkers(ref_params['search_range'],ref_params['contrast_threshold'],
    #                              ref_params['delimitations_list'], ref_params['chosen_frame_add'])
    # marker.choose_folder()
    # marker.process_all_videos()


    # print("choose the _marked folder to continue, filedialog can be hidden behind your IDE")
    #
    # # DataManipulator: Organize and guide features extraction from data
    manipulator = fe.DataManipulator(ref_params['delimitation_flags'], ref_params['delimitations_list'],
                                     video_params['trial_dict'], video_params['trial_dict_2'])
    manipulator.choose_folder()
    #
    # # Calculator: Compute and extract features
    features_calculator = fe.Calculator()
    features_calculator.get_parameters(res_params["axis_rotation_threshold"], res_params["radian_range"],
                                       res_params['ball_radius'], res_params['column_mapping'])
    manipulator.apply_action_to_all_data(features_calculator.calculate)
    #
    # # Plotter: Plot the extracted features
    harry_plotter = ps.Plotter()
    harry_plotter.get_parameters(plots_params['color_palette'], plots_params["features_to_plot"], video_params['trial_dict'], video_params['trial_dict_2'])
    manipulator.apply_action_to_all_data(harry_plotter.apply_action_on_results)
    manipulator.apply_action_to_all_data(harry_plotter.apply_footprint)
    manipulator.apply_action_to_all_data(harry_plotter.apply_lateral_mean_deduction)
    harry_plotter.finish_and_plot()
    #
    # # StatisticsAnalyzer: Perform statistical analyses
    static_tavori = ps.StatisticsAnalyzer()
    static_tavori.get_parameters(plots_params["features_to_plot"], video_params['trial_dict'])
    manipulator.apply_action_to_all_data(static_tavori.apply_action)
    static_tavori.finish_and_analyze()
