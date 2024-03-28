import cv2
import numpy as np
import glob
import random
import os

import cv2
import numpy as np
import random
from tqdm import tqdm

def apply_effect(image, out, fps=24, duration=2, pre_frame=None):
    height, width = image.shape[:2]
    steps = fps * duration
    effect_type = random.choice(['pan'])
    direction = random.choice(['left_to_right', 'right_to_left', 'low_to_high', 'high_to_low'])
    max_shift_x = width // 15
    max_shift_y = height // 15
    last_frame = image
    applied_trans = False
    if effect_type == 'pan':
        for step in range(steps):
            if step >=0 and step<=int(steps * 0.3):
                progress = step / (steps - 1)  # Normalize to [0, 1]
                if direction == 'left_to_right':
                    shift_x = int(max_shift_x * progress)
                elif direction == 'right_to_left':
                    shift_x = int(max_shift_x * (1 - progress))
                else:
                    shift_x = 0        
                if direction == 'low_to_high':
                    shift_y = int(max_shift_y * progress)
                elif direction == 'high_to_low':
                    shift_y = int(max_shift_y * (1 - progress))
                else:
                    shift_y = 0

                # Calculate new cropping window while preserving the center
                start_x = max(min(shift_x, max_shift_x), 0)
                end_x = start_x + (width - max_shift_x)
                start_y = max(min(shift_y, max_shift_y), 0)
                end_y = start_y + (height - max_shift_y)

                cropped_image = image[start_y:end_y, start_x:end_x]
                frame_resized = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
                if(pre_frame is not None and applied_trans == False):
                    fade_transition(pre_frame, frame_resized, out, duration=1, fps=fps)
                    applied_trans = True
                if step%1==0:
                    out.write(frame_resized)
                last_frame = frame_resized
            elif step >=int(steps * 0.3)+1 and step<=int(steps * 0.6):
                progress = step / (steps - 1)  # Normalize to [0, 1]
                if direction == 'left_to_right':
                    shift_x = int(max_shift_x * progress)
                elif direction == 'right_to_left':
                    shift_x = int(max_shift_x * (1 - progress))
                else:
                    shift_x = 0        
                if direction == 'low_to_high':
                    shift_y = int(max_shift_y * progress)
                elif direction == 'high_to_low':
                    shift_y = int(max_shift_y * (1 - progress))
                else:
                    shift_y = 0

                # Calculate new cropping window while preserving the center
                start_x = max(min(shift_x, max_shift_x), 0)
                end_x = start_x + (width - max_shift_x)
                start_y = max(min(shift_y, max_shift_y), 0)
                end_y = start_y + (height - max_shift_y)

                cropped_image = image[start_y:end_y, start_x:end_x]
                frame_resized = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
                if(pre_frame is not None and applied_trans == False):
                    fade_transition(pre_frame, frame_resized, out, duration=1, fps=fps)
                    applied_trans = True
                if step%1==0:
                    out.write(frame_resized)
                last_frame = frame_resized
            elif step >=int(steps * 0.6)+1 and step<steps:
                progress = step / (steps - 1)  # Normalize to [0, 1]
                if direction == 'left_to_right':
                    shift_x = int(max_shift_x * progress)
                elif direction == 'right_to_left':
                    shift_x = int(max_shift_x * (1 - progress))
                else:
                    shift_x = 0        
                if direction == 'low_to_high':
                    shift_y = int(max_shift_y * progress)
                elif direction == 'high_to_low':
                    shift_y = int(max_shift_y * (1 - progress))
                else:
                    shift_y = 0

                # Calculate new cropping window while preserving the center
                start_x = max(min(shift_x, max_shift_x), 0)
                end_x = start_x + (width - max_shift_x)
                start_y = max(min(shift_y, max_shift_y), 0)
                end_y = start_y + (height - max_shift_y)

                cropped_image = image[start_y:end_y, start_x:end_x]
                frame_resized = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
                if(pre_frame is not None and applied_trans == False):
                    fade_transition(pre_frame, frame_resized, out, duration=1, fps=fps)
                    applied_trans = True
                out.write(frame_resized)
                last_frame = frame_resized
            else:
                print("ERROR IN FRAGMENTATION.")
        return last_frame

    elif effect_type == 'zoom':
        zoom_factor = 1.1
        zoom_scales = np.linspace(1, zoom_factor, steps) if direction in ['left_to_right', 'low_to_high'] else np.linspace(zoom_factor, 1, steps)

        for scale in zoom_scales:
        # Resize the image
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
            # Calculate the crop dimensions
            start_x = max(0, resized.shape[1] // 2 - width // 2)
            end_x = start_x + width
            start_y = max(0, resized.shape[0] // 2 - height // 2)
            end_y = start_y + height

            # Ensure the crop dimensions are within the resized image bounds
            end_x = min(end_x, resized.shape[1])
            end_y = min(end_y, resized.shape[0])

            # Crop the resized image
            frame = resized[start_y:end_y, start_x:end_x]
            if(pre_frame is not None and applied_trans == False):
                fade_transition(pre_frame, frame, out, duration=1, fps=fps)
                applied_trans = True
            out.write(frame)
            last_frame = frame
        return last_frame

def fade_transition(prev_image, next_image, out, duration=1, fps=24):
    ignored_frame = 0
    steps = fps * duration
    for step in range(steps):
        if step>=(ignored_frame-1) and step<=(steps-ignored_frame):
            if step%8==0:
                alpha = step / steps
                frame = cv2.addWeighted(prev_image, 1 - alpha, next_image, alpha, 0)
                out.write(frame)

def resize_and_crop(img, target_width=1280, target_height=720):
    original_height, original_width = img.shape[:2]
    
    # Calculate the scaling factors for width and height independently
    scaling_factor_width = target_width / original_width
    scaling_factor_height = target_height / original_height
    
    # Choose the larger scaling factor to ensure the image covers the target dimensions
    scaling_factor = max(scaling_factor_width, scaling_factor_height)
    
    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    
    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Calculate cropping positions
    x_crop = max(0, new_width - target_width) // 2
    y_crop = max(0, new_height - target_height) // 2
    
    # Crop the image
    cropped_img = resized_img[y_crop:y_crop+target_height, x_crop:x_crop+target_width]
    
    return cropped_img

def stabilize_video(input_path, output_path):
    # Capture the input video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read the first frame
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Pre-defined transformation matrix
    transforms = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3), np.float32)

    for i in range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Capture the next frame
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Estimate motion between frames
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Estimate transformation matrix
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

    # Accumulate the transformations to apply smoothing
    for i in range(1, len(transforms)):
        transforms[i, 0] += transforms[i-1, 0]  # dx
        transforms[i, 1] += transforms[i-1, 1]  # dy
        transforms[i, 2] += transforms[i-1, 2]  # da

    # Apply smoothing using a moving average filter
    radius = 15  # Adjust this value based on your video
    kernel = np.ones(radius) / radius
    transforms[:, 0] = np.convolve(transforms[:, 0], kernel, mode='same')
    transforms[:, 1] = np.convolve(transforms[:, 1], kernel, mode='same')
    transforms[:, 2] = np.convolve(transforms[:, 2], kernel, mode='same')

    # Reset video to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Apply stabilized transformations
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        # Compute the transformation matrix
        dx = transforms[i, 0]
        dy = transforms[i, 1]
        da = transforms[i, 2]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        stabilized_frame = cv2.warpAffine(frame, m, (width, height))

        # Write the stabilized frame
        out.write(stabilized_frame)

    # Release video objects
    cap.release()
    out.release()
    return output_path
frame_width = 1920
frame_height = 1080
fps = 144
size = (frame_width, frame_height)

out = cv2.VideoWriter('video_with_effects.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

folder_path = "Assets/property-1/"  # Make sure to change this to your folder path
image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))


prev_img = None
for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing Images"):
    img = cv2.imread(img_path)
    img = resize_and_crop(img, frame_width, frame_height)
    # Apply pan or zoom effect to the current image
    if i==0:
        prev_img = apply_effect(img, out, fps=fps, duration=3, pre_frame=None)
    else:
        prev_img = apply_effect(img, out, fps=fps, duration=3, pre_frame=prev_img)
   

out.release()

print("Video has been created successfully.")
#print("Starting enhancing.")
#stabilize_video("video_with_effects.mp4","final.mp4")
#print("Enhancing Done.")
