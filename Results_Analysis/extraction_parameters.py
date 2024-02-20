P = {
    "video_processor_params": {
        'trial_dict': {"OPEN-LOOP": (1, "O"), "IN-PHASE": (2, "I"), "OUT-OF-PHASE": (3, "P"), "ALTERNATE": (4, "A")}, # "ALTERNATE": (4, "A")
        'trial_dict_2': {'FORWARD': (1, 'F'), 'BACKWARD': (2, 'B')},
        'experiment_name': 'Peer recognition experiment',
        'convert_params': ['.MP4', '.avi', '25', '1280x720', '5000k'],  # source extension, format to convert, frames, resolution, bitmaps
        'start_num_video': 1,
        'fictrac_directory': "C://Users/scr/vcpkg/fictrac/bin/Release",
        'formatted_name': "\d{1,3}_[A-Za-z]_[A-Za-z]_\d{1,3}\.\w+",  # regular expression to format the names
        'expected_size_file_bits': (1000000000, 4000000000)
    },
    "reference_point_params": {
            "delimitation_flags": [(111, "black_screen"), (222, "pre_trial"), (999, "trial"), (0, "end_of_the_file")],
            "delimitations_list": [(111, -3000), (222, -1500), (999,  0), (000, 1000)],
            'search_range': (1300, 1800),
            'contrast_threshold': 10,
            'chosen_frame_add': 1500  # index to add to the contrast chosen frame
    },
    "results_extraction": {
            "ball_radius": 6,
            "axis_rotation_threshold": [0.004, 0.004, 0.003],
            "radian_range": 0.52,  # not used currently
            "column_mapping": {'5': 'x', '7': 'z', '14': 'dis_x', '15': 'dis_y'}
    },
    "plots_and_stats": {
        "color_palette": [(218 / 255, 207 / 255, 79 / 255), (192 / 255, 0 / 255, 0 / 255)],
        "features_to_plot": ['walking_fraction', 'average_pause_duration [s]', r'relative_speed [cm.s-1]', 'number_of_pauses',
                             'lateral_proportion_25_degree', 'lateral_proportion_45_degree', 'distance_walked [cm]'],
        'paired_subjects': True
    }

}
