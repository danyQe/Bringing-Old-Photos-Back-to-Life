try:
    import cv2
    print("OpenCV successfully imported. Version:", cv2.__version__)
except ImportError as e:
    print("ERROR: Failed to import OpenCV (cv2):", str(e))
    print("Please install OpenCV with: pip install opencv-python")
    raise

import numpy as np
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

def setup_colorization_model():
    """Setup and return the colorization model and required parameters"""
    try:
        # Load pre-trained model
        prototxt = "Global/models/colorization_deploy_v2.prototxt"
        caffemodel = "Global/models/colorization_release_v2.caffemodel"
        pts_npy = "Global/models/pts_in_hull.npy"

        # Set up the model
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        pts = np.load(pts_npy)
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype(np.float32)]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        return net
    except Exception as e:
        logger.error(f"Error setting up colorization model: {str(e)}")
        raise

def colorize_image(input_path, output_path=None):
    """
    Colorize a restored grayscale image using the OpenCV DNN colorization model.
    
    Args:
        input_path (str): Path to the input image
        output_path (str, optional): Path to save the colorized output. If None,
                                     will save as 'colorized_' + input filename in the same directory.
    
    Returns:
        str: Path to the colorized image
    """
    try:
        # Setup the path to save the colorized image if not provided
        if output_path is None:
            dir_path = os.path.dirname(input_path)
            filename = os.path.basename(input_path)
            output_path = os.path.join(dir_path, f"colorized_{filename}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Colorizing image: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # Setup the model
        net = setup_colorization_model()
        
        # Load image in BGR format (OpenCV default)
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read input image: {input_path}")
        
        # Store original dimensions
        original_h, original_w = image.shape[:2]
        
        # Convert to float32 and normalize
        scaled = image.astype(np.float32) / 255.0
        
        # Convert BGR to Lab color space (consistent with OpenCV's BGR format)
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        
        # Resize for the model input (224x224)
        lab_resized = cv2.resize(lab, (224, 224))
        
        # Extract L channel and apply mean subtraction
        L_resized = lab_resized[:,:,0]
        L_resized -= 50
        
        # Predict ab channels
        net.setInput(cv2.dnn.blobFromImage(L_resized))
        ab_predicted = net.forward()[0,:,:,:].transpose((1, 2, 0))

        # Resize prediction back to original size
        ab_resized = cv2.resize(ab_predicted, (original_w, original_h))
        
        # Get original L channel (not resized)
        L_original = lab[:,:,0]
        
        # Combine original L channel with predicted ab channels
        colorized_lab = np.concatenate((L_original[:,:,np.newaxis], ab_resized), axis=2)

        # Convert back to BGR color space
        colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        
        # Ensure values are in valid range
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        
        # Apply subtle color balance correction to reduce any color cast
        # This helps neutralize any remaining tint issues
        colorized_bgr[:,:,0] = colorized_bgr[:,:,0] * 0.95  # Slightly reduce blue channel
        colorized_bgr[:,:,2] = colorized_bgr[:,:,2] * 0.98  # Slightly reduce red channel
        
        # Clip again after adjustment
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        
        # Convert to uint8 for saving
        colorized_final = (colorized_bgr * 255).astype(np.uint8)

        # Save result (cv2.imwrite expects BGR format, which we have)
        cv2.imwrite(output_path, colorized_final)
        logger.info(f"Colorized image saved to: {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error colorizing image: {str(e)}")
        # If colorization fails, return the original image path
        logger.info(f"Returning original image due to colorization failure")
        return input_path

def batch_colorize_directory(input_dir, output_dir=None):
    """
    Colorize all images in a directory
    
    Args:
        input_dir (str): Directory containing images to colorize
        output_dir (str, optional): Directory to save colorized images. If None,
                                   will create a 'colorized' subdirectory in input_dir.
    
    Returns:
        list: Paths to all colorized images
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "colorized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    colorized_paths = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            try:
                colorized_path = colorize_image(input_path, output_path)
                colorized_paths.append(colorized_path)
            except Exception as e:
                logger.error(f"Error colorizing {filename}: {str(e)}")
                # Add the original path if colorization fails
                colorized_paths.append(input_path)
    
    return colorized_paths
