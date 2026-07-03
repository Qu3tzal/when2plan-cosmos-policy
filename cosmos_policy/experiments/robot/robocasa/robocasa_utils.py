# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in RoboCasa simulation environments."""

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cosmos_policy.experiments.robot.robot_utils import DATE_TIME


def save_rollout_video(
    rollout_primary_images,
    rollout_secondary_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    rollout_data_dir,
    log_file=None,
):
    """Saves an MP4 replay of an episode with all three camera views."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
    mp4_path = (
        f"{rollout_data_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Concatenate all three camera views horizontally: primary (left) | secondary (right) | wrist
    for primary_img, secondary_img, wrist_img in zip(
        rollout_primary_images, rollout_secondary_images, rollout_wrist_images
    ):
        combined_img = np.concatenate([primary_img, secondary_img, wrist_img], axis=1)
        video_writer.append_data(combined_img)

    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def render_value_graph(
    values_per_inference_step,
    num_open_loop_steps,
    total_frames,
    current_frame,
    width,
    height,
    font=None,
):
    """Render a line plot of the planned value over the episode with a marker at the current frame.

    The value function outputs one scalar per inference step (best-of-N requery); that value is held
    for `num_open_loop_steps` env frames (a step function). The x-axis is the env timestep, the y-axis
    is the value in [0, 1], and a red vertical line + dot marks where the current frame sits.

    Args:
        values_per_inference_step: list of scalar planned values, one per inference step.
        num_open_loop_steps: env frames each inference step's plan is executed for.
        total_frames: total number of env frames in the episode (x-axis extent).
        current_frame: index of the frame being rendered (marker position).
        width, height: output image size in pixels.
        font: optional PIL font for axis labels.

    Returns:
        np.ndarray (height, width, 3) uint8 image of the graph.
    """
    margin_l, margin_r, margin_t, margin_b = 44, 12, 14, 22
    plot_w = max(1, width - margin_l - margin_r)
    plot_h = max(1, height - margin_t - margin_b)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    vmin, vmax = 0.0, 1.0  # value-function output range

    def to_xy(frame, value):
        if total_frames > 1:
            x = margin_l + plot_w * (frame / (total_frames - 1))
        else:
            x = margin_l
        y = margin_t + plot_h * (1.0 - (value - vmin) / (vmax - vmin))
        return int(round(x)), int(round(y))  # int coords for broad Pillow compatibility

    # Plot border + horizontal gridlines/labels at 0.0, 0.5, 1.0
    draw.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h], outline=(0, 0, 0))
    for yval in (0.0, 0.5, 1.0):
        _, gy = to_xy(0, yval)
        draw.line([(margin_l, gy), (margin_l + plot_w, gy)], fill=(220, 220, 220))
        if font is not None:
            draw.text((4, gy - 6), f"{yval:.1f}", font=font, fill=(0, 0, 0))

    n_steps = len(values_per_inference_step)
    if n_steps == 0 or total_frames <= 0:
        return np.array(img, dtype=np.uint8)

    # Build per-frame value series (step function held over each plan's open-loop steps)
    points = []
    for f in range(total_frames):
        step_idx = min(f // max(1, num_open_loop_steps), n_steps - 1)
        points.append(to_xy(f, values_per_inference_step[step_idx]))
    if len(points) >= 2:
        draw.line(points, fill=(30, 90, 200), width=2)

    # Current-frame marker: vertical line + dot at the current planned value
    cur_frame = min(max(current_frame, 0), total_frames - 1)
    cur_step = min(cur_frame // max(1, num_open_loop_steps), n_steps - 1)
    cur_val = float(values_per_inference_step[cur_step])
    cx, cy = to_xy(cur_frame, cur_val)
    draw.line([(cx, margin_t), (cx, margin_t + plot_h)], fill=(200, 30, 30), width=1)
    r = 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 30, 30))
    if font is not None:
        draw.text(
            (margin_l, margin_t + plot_h + 4),
            f"planned value = {cur_val:.3f}",
            font=font,
            fill=(0, 0, 0),
        )

    return np.array(img, dtype=np.uint8)


def save_rollout_video_with_future_image_predictions(
    rollout_primary_images,
    rollout_secondary_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    rollout_data_dir,
    chunk_size,
    num_open_loop_steps,
    future_primary_image_predictions=None,
    future_secondary_image_predictions=None,
    future_wrist_image_predictions=None,
    value_predictions=None,
    show_diff=False,
    log_file=None,
    show_timestep=False,
    timestep=0,
):
    """Saves an MP4 replay of an episode with 3 rows and 3 columns:
    Top row:    current wrist, current primary, current secondary images
    Middle row: the wrist/primary/secondary images that were fed to the world model as conditioning
                input for the prediction shown below (the observation at the last requery frame).
    Bottom row: future wrist, future primary, future secondary predictions.

    For RoboCasa, we have three camera views:
    - Wrist (eye-in-hand)
    - Primary (left third-person)
    - Secondary (right third-person)

    Args:
        rollout_primary_images: List of primary (left) camera images
        rollout_secondary_images: List of secondary (right) camera images
        rollout_wrist_images: List of wrist camera images
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        chunk_size: Number of timesteps for future prediction
        num_open_loop_steps: Number of open loop steps
        future_primary_image_predictions: List of predicted future primary images
        future_secondary_image_predictions: List of predicted future secondary images
        future_wrist_image_predictions: List of predicted future wrist images
        show_diff: If True, show difference images (ignored in this version)
        log_file: Optional file for logging
        show_timestep: If True, show the timestep on the video
        timestep: The current timestep
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_data_dir}/{DATE_TIME}--with_future_img--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Ensure future prediction lists have at least one element
    if not future_wrist_image_predictions:
        raise ValueError("future_wrist_image_predictions must have at least one element")
    if not future_primary_image_predictions:
        raise ValueError("future_primary_image_predictions must have at least one element")
    if not future_secondary_image_predictions:
        raise ValueError("future_secondary_image_predictions must have at least one element")

    # Get dimensions from future predictions to use for resizing
    target_h, target_w, c = future_primary_image_predictions[0].shape

    # Define text parameters
    text_height = 60 if show_timestep else 30  # Height for text area (increased if showing timestep)
    font_size = 16

    # Define column labels
    column_labels = ["wrist image", "primary image (left)", "secondary image (right)"]

    for i, (primary_img, secondary_img, wrist_img) in enumerate(
        zip(rollout_primary_images, rollout_secondary_images, rollout_wrist_images)
    ):
        # Process current images - resize to match future prediction dimensions
        current_images_to_process = [wrist_img, primary_img, secondary_img]
        processed_current_images = []

        for current_img in current_images_to_process:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(current_img)

            # Resize if needed
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

            # Convert back to numpy array
            processed_current_images.append(np.array(pil_img))

        # Unpack processed current images
        wrist_img_resized, primary_img_resized, secondary_img_resized = processed_current_images

        # Determine which future prediction images to use
        future_idx = i // num_open_loop_steps
        future_wrist_idx = min(future_idx, len(future_wrist_image_predictions) - 1)
        future_primary_idx = min(future_idx, len(future_primary_image_predictions) - 1)
        future_secondary_idx = min(future_idx, len(future_secondary_image_predictions) - 1)

        future_wrist_img = future_wrist_image_predictions[future_wrist_idx]
        future_primary_img = future_primary_image_predictions[future_primary_idx]
        future_secondary_img = future_secondary_image_predictions[future_secondary_idx]

        # Determine the images that were fed to the world model as conditioning input for the prediction
        # shown below. The world model is requeried every `num_open_loop_steps` frames, so inference step
        # `future_idx` conditioned on the observation captured at frame `future_idx * num_open_loop_steps`
        # (clamped to the available rollout / prediction lengths so it stays aligned with the row below).
        wm_input_inference_step = min(future_idx, len(future_primary_image_predictions) - 1)
        wm_input_frame_idx = min(wm_input_inference_step * num_open_loop_steps, len(rollout_primary_images) - 1)
        wm_input_images_to_process = [
            rollout_wrist_images[wm_input_frame_idx],
            rollout_primary_images[wm_input_frame_idx],
            rollout_secondary_images[wm_input_frame_idx],
        ]
        processed_wm_input_images = []
        for wm_input_img in wm_input_images_to_process:
            pil_img = Image.fromarray(wm_input_img)
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
            processed_wm_input_images.append(np.array(pil_img))
        wm_input_wrist_resized, wm_input_primary_resized, wm_input_secondary_resized = processed_wm_input_images

        # Create a combined image with 3 rows and 3 columns
        combined_img = np.zeros((target_h * 3, target_w * 3, c), dtype=np.uint8)

        # Top row: current images (wrist, primary, secondary)
        combined_img[:target_h, :target_w, :] = wrist_img_resized
        combined_img[:target_h, target_w : target_w * 2, :] = primary_img_resized
        combined_img[:target_h, target_w * 2 : target_w * 3, :] = secondary_img_resized

        # Middle row: world model input images (the conditioning observation for this inference step)
        combined_img[target_h : target_h * 2, :target_w, :] = wm_input_wrist_resized
        combined_img[target_h : target_h * 2, target_w : target_w * 2, :] = wm_input_primary_resized
        combined_img[target_h : target_h * 2, target_w * 2 : target_w * 3, :] = wm_input_secondary_resized

        # Bottom row: future predictions (wrist, primary, secondary)
        combined_img[target_h * 2 :, :target_w, :] = future_wrist_img
        combined_img[target_h * 2 :, target_w : target_w * 2, :] = future_primary_img
        combined_img[target_h * 2 :, target_w * 2 : target_w * 3, :] = future_secondary_img

        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * 3, 3), dtype=np.uint8) * 255

        # Convert numpy array to PIL Image for text drawing
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)

        # Try to use a standard font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Verdana", font_size)
                except IOError:
                    font = ImageFont.load_default()

        # Add timestep if requested
        if show_timestep:
            timestep_text = f"t = {i}"
            timestep_width = draw.textlength(timestep_text, font=font)
            # Draw timestep centered at the top
            draw.text(((target_w * 3 - timestep_width) // 2, 2), timestep_text, font=font, fill=(0, 0, 0))

        # Add column labels
        label_y_pos = 32 if show_timestep else 8  # Adjust y position based on whether timestep is shown
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, label_y_pos), label, font=font, fill=(0, 0, 0))

        # Convert back to numpy array
        text_area = np.array(text_img)

        # Label each image row so the three near-identical rows are easy to tell apart.
        combined_pil = Image.fromarray(combined_img)
        combined_draw = ImageDraw.Draw(combined_pil)
        row_labels = ["current obs", "world model input", "world model output"]
        for row_idx, row_label in enumerate(row_labels):
            combined_draw.text((4, row_idx * target_h + 4), row_label, font=font, fill=(255, 255, 0))
        combined_img = np.array(combined_pil)

        # Combine text area and images
        final_frame = np.vstack((text_area, combined_img))

        # Optional 4th row: planned-value graph over time with a marker at the current frame
        if value_predictions is not None and len(value_predictions) > 0:
            graph_row = render_value_graph(
                values_per_inference_step=value_predictions,
                num_open_loop_steps=num_open_loop_steps,
                total_frames=len(rollout_primary_images),
                current_frame=i,
                width=target_w * 3,
                height=target_h,
                font=font,
            )
            final_frame = np.vstack((final_frame, graph_row))

        video_writer.append_data(final_frame)

    video_writer.close()
    print(f"Saved rollout MP4 with future predictions at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 with future predictions at path {mp4_path}\n")
    return mp4_path


def save_rollout_video_with_future_image_predictions_and_gt(
    rollout_primary_images,
    rollout_secondary_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    rollout_data_dir,
    chunk_size,
    num_open_loop_steps,
    future_primary_image_predictions=None,
    future_secondary_image_predictions=None,
    future_wrist_image_predictions=None,
    gt_future_primary_image_predictions=None,
    gt_future_secondary_image_predictions=None,
    gt_future_wrist_image_predictions=None,
    show_diff=True,
    log_file=None,
    show_timestep=False,
    timestep=0,
):
    """Saves an MP4 replay of an episode with 2 rows and 3 columns:
    Top row: current wrist, current primary, current secondary images
    Bottom row: future wrist, future primary, future secondary predictions.

    For RoboCasa, we have three camera views:
    - Wrist (eye-in-hand)
    - Primary (left third-person)
    - Secondary (right third-person)

    Args:
        rollout_primary_images: List of primary (left) camera images
        rollout_secondary_images: List of secondary (right) camera images
        rollout_wrist_images: List of wrist camera images
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        chunk_size: Number of timesteps for future prediction
        num_open_loop_steps: Number of open loop steps
        future_primary_image_predictions: List of predicted future primary images
        future_secondary_image_predictions: List of predicted future secondary images
        future_wrist_image_predictions: List of predicted future wrist images
        show_diff: If True, show difference images (ignored in this version)
        log_file: Optional file for logging
        show_timestep: If True, show the timestep on the video
        timestep: The current timestep
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_data_dir}/{DATE_TIME}--with_future_img--episode={idx}--success={success}--task={processed_task_description}--gt.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Ensure future prediction lists have at least one element
    if not future_wrist_image_predictions:
        raise ValueError("future_wrist_image_predictions must have at least one element")
    if not future_primary_image_predictions:
        raise ValueError("future_primary_image_predictions must have at least one element")
    if not future_secondary_image_predictions:
        raise ValueError("future_secondary_image_predictions must have at least one element")

    # Get dimensions from future predictions to use for resizing
    target_h, target_w, c = future_primary_image_predictions[0].shape

    # Define text parameters
    text_height = 60 if show_timestep else 30  # Height for text area (increased if showing timestep)
    font_size = 16

    # Define column labels
    column_labels = ["wrist image", "primary image (left)", "secondary image (right)"]

    # Center-crop the ground-truth future images to match the future prediction images
    def center_crop(img, img_size):
        import torch
        import torchvision.transforms.functional as F

        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
        crop_size = int(img_size * 0.9**0.5)  # Square root because we're dealing with area
        img_crop = F.center_crop(img_tensor, crop_size)
        img_resized = F.resize(img_crop, [img_size, img_size], antialias=True)
        return img_resized.numpy().transpose(1, 2, 0)

    gt_future_wrist_images = []
    gt_future_primary_images = []
    gt_future_secondary_images = []
    for gt_future_wrist_img, gt_future_primary_img, gt_future_secondary_img in zip(
        gt_future_wrist_image_predictions, gt_future_primary_image_predictions, gt_future_secondary_image_predictions
    ):
        gt_future_wrist_img = center_crop(gt_future_wrist_img, target_h)
        gt_future_primary_img = center_crop(gt_future_primary_img, target_h)
        gt_future_secondary_img = center_crop(gt_future_secondary_img, target_h)
        gt_future_wrist_images.append(gt_future_wrist_img)
        gt_future_primary_images.append(gt_future_primary_img)
        gt_future_secondary_images.append(gt_future_secondary_img)
    gt_future_wrist_image_predictions = gt_future_wrist_images
    gt_future_primary_image_predictions = gt_future_primary_images
    gt_future_secondary_image_predictions = gt_future_secondary_images

    for i, (primary_img, secondary_img, wrist_img) in enumerate(
        zip(rollout_primary_images, rollout_secondary_images, rollout_wrist_images)
    ):
        # Process current images - resize to match future prediction dimensions
        current_images_to_process = [wrist_img, primary_img, secondary_img]
        processed_current_images = []

        for current_img in current_images_to_process:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(current_img)

            # Resize if needed
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

            # Convert back to numpy array
            processed_current_images.append(np.array(pil_img))

        # Unpack processed current images
        wrist_img_resized, primary_img_resized, secondary_img_resized = processed_current_images

        # Determine which future prediction images to use
        future_idx = i // num_open_loop_steps
        future_wrist_idx = min(future_idx, len(future_wrist_image_predictions) - 1)
        future_primary_idx = min(future_idx, len(future_primary_image_predictions) - 1)
        future_secondary_idx = min(future_idx, len(future_secondary_image_predictions) - 1)

        future_wrist_img = future_wrist_image_predictions[future_wrist_idx]
        future_primary_img = future_primary_image_predictions[future_primary_idx]
        future_secondary_img = future_secondary_image_predictions[future_secondary_idx]

        gt_future_wrist_img = gt_future_wrist_image_predictions[future_wrist_idx]
        gt_future_primary_img = gt_future_primary_image_predictions[future_primary_idx]
        gt_future_secondary_img = gt_future_secondary_image_predictions[future_secondary_idx]

        # Compute difference images if show_diff is True
        if show_diff:
            # Compute difference between future primary image and ground-truth future primary image
            primary_diff = np.abs(future_primary_img.astype(np.float32) - gt_future_primary_img.astype(np.float32))
            primary_diff = np.clip(primary_diff, 0, 255).astype(np.uint8)
            # Compute difference between future wrist image and ground-truth future wrist image
            wrist_diff = np.abs(future_wrist_img.astype(np.float32) - gt_future_wrist_img.astype(np.float32))
            wrist_diff = np.clip(wrist_diff, 0, 255).astype(np.uint8)
            # Compute difference between future secondary image and ground-truth future secondary image
            secondary_diff = np.abs(
                future_secondary_img.astype(np.float32) - gt_future_secondary_img.astype(np.float32)
            )
            secondary_diff = np.clip(secondary_diff, 0, 255).astype(np.uint8)

        # Create a combined image with 4 rows and 3 columns
        combined_img = np.zeros((target_h * 4, target_w * 3, c), dtype=np.uint8)

        # Top row: current images (wrist, primary, secondary)
        combined_img[:target_h, :target_w, :] = wrist_img_resized
        combined_img[:target_h, target_w : target_w * 2, :] = primary_img_resized
        combined_img[:target_h, target_w * 2 : target_w * 3, :] = secondary_img_resized
        # Second row: future predictions (wrist, primary, secondary)
        combined_img[target_h : target_h * 2, :target_w, :] = future_wrist_img
        combined_img[target_h : target_h * 2, target_w : target_w * 2, :] = future_primary_img
        combined_img[target_h : target_h * 2, target_w * 2 : target_w * 3, :] = future_secondary_img
        # Third row: ground-truth future images (wrist, primary, secondary)
        combined_img[target_h * 2 : target_h * 3, :target_w, :] = gt_future_wrist_img
        combined_img[target_h * 2 : target_h * 3, target_w : target_w * 2, :] = gt_future_primary_img
        combined_img[target_h * 2 : target_h * 3, target_w * 2 : target_w * 3, :] = gt_future_secondary_img
        # Fourth row: difference images (wrist, primary, secondary)
        combined_img[target_h * 3 : target_h * 4, :target_w, :] = wrist_diff
        combined_img[target_h * 3 : target_h * 4, target_w : target_w * 2, :] = primary_diff
        combined_img[target_h * 3 : target_h * 4, target_w * 2 : target_w * 3, :] = secondary_diff
        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * 3, 3), dtype=np.uint8) * 255

        # Convert numpy array to PIL Image for text drawing
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)

        # Try to use a standard font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Verdana", font_size)
                except IOError:
                    font = ImageFont.load_default()

        # Add timestep if requested
        if show_timestep:
            timestep_text = f"t = {i}"
            timestep_width = draw.textlength(timestep_text, font=font)
            # Draw timestep centered at the top
            draw.text(((target_w * 3 - timestep_width) // 2, 2), timestep_text, font=font, fill=(0, 0, 0))

        # Add column labels
        label_y_pos = 32 if show_timestep else 8  # Adjust y position based on whether timestep is shown
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, label_y_pos), label, font=font, fill=(0, 0, 0))

        # Convert back to numpy array
        text_area = np.array(text_img)

        # Combine text area and images
        final_frame = np.vstack((text_area, combined_img))

        video_writer.append_data(final_frame)

    video_writer.close()
    print(f"Saved rollout MP4 with future predictions and ground-truth future images at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 with future predictions and ground-truth future images at path {mp4_path}\n")
    return mp4_path
