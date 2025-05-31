# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# CPU-Optimized and Fixed Face Detection for standard mode

import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from PIL import Image
import torch.nn.functional as F
import torchvision as tv
import torchvision.utils as vutils
import time
import cv2
import os
from skimage import img_as_ubyte
import json
import argparse
import dlib
import logging
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Memory cleanup for face detection"""
    gc.collect()
    memory_info = psutil.virtual_memory()
    if memory_info.percent > 80:
        logger.warning(f"High memory usage: {memory_info.percent}%")

def ensure_image_format_fixed(image):
    """Ensure image is in the correct format for dlib face detection - FIXED VERSION"""
    try:
        # Handle different input types
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image)
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy array or PIL Image, got {type(image)}")
        
        # Ensure we have a valid image
        if len(image.shape) < 2:
            raise ValueError(f"Invalid image shape: {image.shape}")
        
        # Handle different data types
        if image.dtype == np.float64 or image.dtype == np.float32:
            # Convert from [0,1] to [0,255]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            # Convert any other type to uint8
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Handle different channel configurations
        if len(image.shape) == 2:
            # Grayscale - convert to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # RGB - already correct
                pass
            elif image.shape[2] == 1:  # Single channel
                image = np.repeat(image, 3, axis=2)
            else:
                raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unexpected image dimensions: {len(image.shape)}")
        
        # Final validation
        assert image.dtype == np.uint8, f"Final image dtype is {image.dtype}, should be uint8"
        assert len(image.shape) == 3, f"Final image shape is {image.shape}, should be 3D"
        assert image.shape[2] == 3, f"Final image has {image.shape[2]} channels, should be 3"
        assert 0 <= image.min() and image.max() <= 255, f"Image values out of range: [{image.min()}, {image.max()}]"
        
        logger.debug(f"Image format ensured: {image.dtype}, {image.shape}, range [{image.min()}, {image.max()}]")
        return image
        
    except Exception as e:
        logger.error(f"Error ensuring image format: {str(e)}")
        raise

def _standard_face_pts():
    pts = (
        np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32) / 256.0
        - 1.0
    )
    return np.reshape(pts, (5, 2))

def _origin_face_pts():
    pts = np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32)
    return np.reshape(pts, (5, 2))

def get_landmark(face_landmarks, id):
    part = face_landmarks.part(id)
    x = part.x
    y = part.y
    return (x, y)

def search(face_landmarks):
    x1, y1 = get_landmark(face_landmarks, 36)
    x2, y2 = get_landmark(face_landmarks, 39)
    x3, y3 = get_landmark(face_landmarks, 42)
    x4, y4 = get_landmark(face_landmarks, 45)

    x_nose, y_nose = get_landmark(face_landmarks, 30)

    x_left_mouth, y_left_mouth = get_landmark(face_landmarks, 48)
    x_right_mouth, y_right_mouth = get_landmark(face_landmarks, 54)

    x_left_eye = int((x1 + x2) / 2)
    y_left_eye = int((y1 + y2) / 2)
    x_right_eye = int((x3 + x4) / 2)
    y_right_eye = int((y3 + y4) / 2)

    results = np.array([
        [x_left_eye, y_left_eye],
        [x_right_eye, y_right_eye],
        [x_nose, y_nose],
        [x_left_mouth, y_left_mouth],
        [x_right_mouth, y_right_mouth],
    ])

    return results

def compute_transformation_matrix(img, landmark, normalize, target_face_scale=1.0):
    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0  # 256 for standard mode

    h, w, c = img.shape
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    affine = SimilarityTransform()
    affine.estimate(target_pts, landmark)

    return affine.params

def load_and_prepare_image(img_path):
    """Load image and prepare it for face detection"""
    try:
        logger.debug(f"Loading image: {img_path}")
        
        # Load image using multiple methods as fallback
        image = None
        
        # Method 1: PIL Image (recommended)
        try:
            pil_img = Image.open(img_path).convert("RGB")
            image = np.array(pil_img)
            logger.debug(f"Loaded with PIL: {image.shape}, {image.dtype}")
        except Exception as e:
            logger.warning(f"PIL loading failed: {str(e)}")
        
        # Method 2: OpenCV (fallback)
        if image is None:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Loaded with OpenCV: {image.shape}, {image.dtype}")
            except Exception as e:
                logger.warning(f"OpenCV loading failed: {str(e)}")
        
        # Method 3: skimage (last resort)
        if image is None:
            try:
                image = io.imread(img_path)
                logger.debug(f"Loaded with skimage: {image.shape}, {image.dtype}")
            except Exception as e:
                logger.error(f"All loading methods failed for {img_path}: {str(e)}")
                return None
        
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            return None
        
        # Ensure correct format for dlib
        image = ensure_image_format_fixed(image)
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading/preparing image {img_path}: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="input directory")
    parser.add_argument("--save_url", type=str, required=True, help="output directory")
    opts = parser.parse_args()

    url = opts.url
    save_url = opts.save_url

    # Create directories
    os.makedirs(url, exist_ok=True)
    os.makedirs(save_url, exist_ok=True)

    # Initialize face detector and landmark predictor
    try:
        face_detector = dlib.get_frontal_face_detector()
        
        # Try to find landmark predictor
        predictor_paths = [
            "shape_predictor_68_face_landmarks.dat",
            "Face_Detection/shape_predictor_68_face_landmarks.dat",
            os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
        ]
        
        landmark_locator = None
        for pred_path in predictor_paths:
            if os.path.exists(pred_path):
                landmark_locator = dlib.shape_predictor(pred_path)
                logger.info(f"Using landmark predictor: {pred_path}")
                break
        
        if landmark_locator is None:
            logger.error("Face landmark predictor not found!")
            logger.error("Please download shape_predictor_68_face_landmarks.dat")
            exit(1)
            
    except Exception as e:
        logger.error(f"Error initializing dlib: {str(e)}")
        exit(1)

    count = 0
    successful_detections = 0
    total_faces_detected = 0

    # Get list of image files
    image_files = []
    for x in os.listdir(url):
        if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(x)
    
    logger.info(f"Found {len(image_files)} image files to process")

    # Process each image
    for x in image_files:
        try:
            img_url = os.path.join(url, x)
            logger.info(f"Processing: {x}")

            # Load and prepare image
            image = load_and_prepare_image(img_url)
            if image is None:
                logger.error(f"Failed to load image: {x}")
                continue

            # Detect faces
            start = time.time()
            try:
                faces = face_detector(image)
                done = time.time()
                logger.debug(f"Face detection took {done - start:.2f} seconds")
            except Exception as e:
                logger.error(f"Face detection failed for {x}: {str(e)}")
                logger.error(f"Image info: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
                continue

            if len(faces) == 0:
                logger.warning(f"No faces detected in {x}")
                continue

            logger.info(f"Found {len(faces)} faces in {x}")
            successful_detections += 1
            total_faces_detected += len(faces)

            # Process each detected face
            for face_id in range(len(faces)):
                try:
                    current_face = faces[face_id]
                    face_landmarks = landmark_locator(image, current_face)
                    current_fl = search(face_landmarks)

                    affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3)
                    aligned_face = warp(image, affine, output_shape=(256, 256, 3))  # 256x256 for standard mode
                    
                    img_name = x[:-4] + "_" + str(face_id + 1)
                    output_path = os.path.join(save_url, img_name + ".png")
                    
                    io.imsave(output_path, img_as_ubyte(aligned_face))
                    logger.debug(f"Saved face: {output_path}")

                except Exception as e:
                    logger.error(f"Error processing face {face_id} in {x}: {str(e)}")
                    continue

            count += 1
            
            # Memory cleanup every 10 images
            if count % 10 == 0:
                cleanup_memory()
                logger.info(f"Processed {count}/{len(image_files)} images...")

        except Exception as e:
            logger.error(f"Error processing {x}: {str(e)}")
            continue

    # Final report
    logger.info("="*60)
    logger.info("FACE DETECTION COMPLETED")
    logger.info("="*60) 
    logger.info(f"Total images processed: {count}")
    logger.info(f"Images with faces detected: {successful_detections}")
    logger.info(f"Total faces extracted: {total_faces_detected}")
    logger.info(f"Results saved in: {save_url}")
    
    if successful_detections == 0:
        logger.warning("No faces were detected in any images!")
        logger.warning("This might indicate:")
        logger.warning("1. Images don't contain clear faces")
        logger.warning("2. Image quality is too low")
        logger.warning("3. Faces are too small or at odd angles")
