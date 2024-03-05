from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN

"""video trimmering"""
# from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips
#
# # Load the video
# video = VideoFileClip(r"D:\videos_locusts\crop.mp4")
#
# # First segment: normal video from 0 to 14 seconds
# first_segment = video.subclip(0, 14)
#
# # Second segment: from 7 to 14 seconds and then 0 to 7 seconds
# second_segment_part1 = video.subclip(7, 14)
# second_segment_part2 = video.subclip(0, 7)
# second_segment = concatenate_videoclips([second_segment_part1, second_segment_part2])
#
# # Resize videos to half their width
# first_segment_resized = first_segment.resize(width=first_segment.size[0]/2)
# second_segment_resized = second_segment.resize(width=second_segment.size[0]/2)
#
# # Create an array with the two segments side by side
# final_clip = clips_array([[first_segment_resized, second_segment_resized]])
#
# # Write the result to a file
# final_clip.write_videofile(r'D:\videos_locusts\unsynchronized_trimmed_crop.mp4', codec='libx264', audio=False)

def cut_video(video_path, duration_seconds):
    """
    Cuts the video to the specified duration and saves the output in the same directory.

    Parameters:
    - video_path: The path to the video file.
    - duration_seconds: The duration to cut the video to, in seconds.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame limit based on duration and fps
    frame_limit = int(duration_seconds * fps)

    # Prepare output path
    video_dir = os.path.dirname(video_path)
    output_filename = "cut_video.avi"
    output_path = os.path.join(video_dir, output_filename)

    # Setup VideoWriter to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0

    while cap.isOpened() and frame_count < frame_limit:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video has been cut and saved to {output_path}")

def save_fgmask_video(video_path, background_video_path):
    # Open the original video and the background video
    cap_video = cv2.VideoCapture(video_path)
    cap_background = cv2.VideoCapture(background_video_path)

    # Retrieve video properties
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output path for the fgmask video
    output_path = os.path.splitext(video_path)[0] + "_fgmask.avi"

    # Define the codec and create a VideoWriter object to write the fgmask video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    while True:
        # Read a frame from both the video and the background video
        ret_video, frame_video = cap_video.read()
        ret_background, frame_background = cap_background.read()

        # Break the loop if video reading is done
        if not ret_video or not ret_background:
            break

        fgmask = cv2.absdiff(frame_video, frame_background)

        # Convert the fgmask to grayscale
        fgmask_gray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)

        # Apply dynamic processing to the fgmask
        fgmask_dynamic = apply_dynamic_processing(fgmask_gray, height)

        # Write the dynamically processed fgmask frame to the output video
        out.write(fgmask_dynamic)

    # Release video resources and save the output
    cap_video.release()
    cap_background.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Foreground mask video saved to {output_path}")

def extract_and_show_background(video_path, return_result=False, iterations=10):
    # Initialize the background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Prepare to write the background video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    background_video_path = os.path.splitext(video_path)[0] + "_background.avi"
    out = cv2.VideoWriter(background_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Iteratively process the video to refine the background model
    for _ in range(iterations):
        # Reset the video capture for each iteration
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no more frames

            # Apply the background subtractor to the part of the frame above 80 pixels
            backSub.apply(frame[80:, :], learningRate=0.01)

    # After refining the model, create the background video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames

        # Extract the background for the part above 80 pixels
        background = backSub.getBackgroundImage()
        if background is not None:
            # Combine the original part below 80 pixels with the background part above 80 pixels
            combined_frame = np.vstack((frame[:80, :], background))

            # Write the combined frame to the output video
            out.write(combined_frame)

    # Clean up
    cap.release()
    out.release()

    print(f"Background video saved to {background_video_path}")

    if return_result:
        # Optionally return the path to the created background video
        return background_video_path
    else:
        return None

def comparing_background_subtraction(video_path, background):
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    grayscale_output_filename = f"{video_name}_grayscale_diff{video_ext}"
    color_output_filename = f"{video_name}_color_diff{video_ext}"
    grayscale_output_path = os.path.join(video_dir, grayscale_output_filename)
    color_output_path = os.path.join(video_dir, color_output_filename)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_gray = cv2.VideoWriter(grayscale_output_path, fourcc, fps, (width, height), isColor=True)
    out_color = cv2.VideoWriter(color_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame-background difference for grayscale
        fgmask_gray = cv2.absdiff(frame, background)
        fgmask_gray = cv2.cvtColor(fgmask_gray, cv2.COLOR_BGR2GRAY)
        _, fgmask_gray_thresh = cv2.threshold(fgmask_gray, 65, 255, cv2.THRESH_BINARY)
        fgmask_gray_bgr = cv2.cvtColor(fgmask_gray_thresh, cv2.COLOR_GRAY2BGR)  # Convert to BGR for video writing
        out_gray.write(fgmask_gray_bgr)

        # Frame-background difference for color with thresholding
        fgmask_color = cv2.absdiff(frame, background)
        fgmask_color_thresh = cv2.inRange(fgmask_color, (40, 40, 40), (255, 255, 255))
        fgmask_color_vis = cv2.bitwise_and(frame, frame, mask=fgmask_color_thresh)
        out_color.write(fgmask_color_vis)

    cap.release()
    out_gray.release()
    out_color.release()

    print(f"Grayscale difference video saved to: {grayscale_output_path}")
    print(f"Color difference video saved to: {color_output_path}")

def draw_coordinates_on_background(background, interval=100):
    # Draw coordinates at a specified interval across the background
    h, w = background.shape[:2]
    for y in range(0, h, interval):
        for x in range(0, w, interval):
            cv2.putText(background, f'({x},{y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.2, (255, 255, 255), 2, cv2.LINE_AA)

def draw_coordinates_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output_path = os.path.splitext(video_path)[0] + "_with_coordinates.avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw coordinates on the frame
        for y in range(0, height, 100):
            for x in range(0, width, 100):
                cv2.putText(frame, f'({x},{y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, height, 90):
            cv2.line(frame, (0, y), (width, y), (255, 0, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
    return output_path

def apply_dynamic_processing(fgmask_gray, height):
    # Initialize the result mask with zeros
    result_mask = np.zeros_like(fgmask_gray)

    # Loop through each row from the specified start Y-coordinate to the bottom of the image
    start_y = 80  # Starting point for dynamic processing
    for y in range(start_y, height):
        # Calculate dynamic parameters based on Y-coordinate
        factor = ((y - start_y) / (height - start_y))
        blur_kernel_size = max(5, int(5 + factor * 15)) | 1  # Ensure the kernel size is odd
        morph_kernel_size = max(3, int(3 + factor * 17)) | 1  # Ensure the kernel size is odd

        # Apply Gaussian Blur to the entire image with dynamic kernel size
        fgmask_blur = cv2.GaussianBlur(fgmask_gray, (blur_kernel_size, blur_kernel_size), 0)

        # Apply threshold
        _, fgmask_thresh = cv2.threshold(fgmask_blur, 25, 255, cv2.THRESH_BINARY)

        # Apply dynamic morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        fgmask_opened = cv2.morphologyEx(fgmask_thresh, cv2.MORPH_OPEN, kernel)
        fgmask_cleaned = cv2.morphologyEx(fgmask_opened, cv2.MORPH_CLOSE, kernel)

        # Update the result mask
        result_mask[y, :] = fgmask_cleaned[y, :]

    return result_mask


def scale_radius(y):
    if y < 85:
        return 1  # Minimum size
    elif y > 230:
        return 90  # Maximum size roughly
    else:
        return int(1 + (y - 85) * (60 - 1) / (230 - 85))


def scale_area_threshold(y, height):
    if y < 90:
        return 10
    else:
        return 18*y-1760  # threshold is 10 at 85 and 3000 at 230

    # return scaled_threshold

def find_centroids(fgmask, height, previous_centroids, distance_threshold=10):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_centroids = []  # To store tuples of (centroid_x, centroid_y, radius)

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            # Initial assumption: the centroid will not be included
            include_centroid = False
            calculated_radius = scale_radius(cY)  # Calculate the radius based on the current centroid's y-coordinate
            for prev_cX, prev_cY, prev_radius in previous_centroids:
                # If the current centroid is close to a previous one, it will be included regardless of its area
                if np.linalg.norm(np.array([cX, cY]) - np.array([prev_cX, prev_cY])) < distance_threshold:
                    new_centroids.append((cX, cY, prev_radius))  # Use the previous radius
                    include_centroid = True
                    break  # Stop checking other centroids once a close one is found
            # If the centroid is not close to any previous ones, check against the scale threshold
            if not include_centroid and cv2.contourArea(contour) > scale_area_threshold(cY, height):
                new_centroids.append((cX, cY, calculated_radius))  # Use the calculated radius

    return new_centroids


def replace_objects_with_dots(video_path, background_video_path):
    previous_centroids = []  # Format: [(centroid_x, centroid_y, radius)]

    cap_video = cv2.VideoCapture(video_path)
    cap_background = cv2.VideoCapture(background_video_path)

    fps = cap_video.get(cv2.CAP_PROP_FPS)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.splitext(video_path)[0] + "_dots.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret_video, frame_video = cap_video.read()
        ret_background, frame_background = cap_background.read()

        if not ret_video or not ret_background:
            break

        fgmask = cv2.absdiff(frame_video, frame_background)
        fgmask_gray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        fgmask = apply_dynamic_processing(fgmask_gray, height)  # Assuming this is a custom function you've defined

        centroids = find_centroids(fgmask, height, previous_centroids)
        background_copy = frame_background.copy()

        for (cX, cY, radius) in centroids:
            cv2.circle(background_copy, (cX, cY), radius, (0, 0, 0), -1)  # Draw with adjusted or previous radius

        previous_centroids = centroids.copy()
        out.write(background_copy)

    cap_video.release()
    cap_background.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output saved to {output_path}")

# Prevent the root window from appearing
Tk().withdraw()

# Open a dialog to choose a file
# video_path = askopenfilename()  # video path

# print(f'Selected file: {video_path}')
# background_video_path = extract_and_show_background(video_path, True)
video_path = fr"C:\Users\yfant\OneDrive\Desktop\cut_video.avi"
background_video_path = fr"C:\Users\yfant\OneDrive\Desktop\cut_video_background.avi"
if background_video_path is not None:

    replace_objects_with_dots(video_path, background_video_path)
    # comparing_background_subtraction(videopath,background)
    # save_fgmask_video(video_path, background_video_path)




