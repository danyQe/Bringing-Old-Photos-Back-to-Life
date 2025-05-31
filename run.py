# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Enhanced for CPU processing, memory optimization, and colorization

import os
import argparse
import shutil
import sys
import gc
import torch
import psutil
from subprocess import call, PIPE, Popen
import logging
from colorize import colorize_image
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Clean up memory after each stage"""
    try:
        # Clear Python garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory usage
        memory_info = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory_info.percent}% ({memory_info.used / (1024**3):.2f}GB / {memory_info.total / (1024**3):.2f}GB)")
        
    except Exception as e:
        logger.warning(f"Memory cleanup warning: {str(e)}")

def run_cmd(cmd):
    """Run a command and handle errors"""
    try:
        # Use subprocess.run with shell=True to handle paths with spaces
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")

def setup_directories(output_folder):
    """Setup and return all required directories"""
    dirs = {
        'stage_1': os.path.join(output_folder, "stage_1_restore_output"),
        'stage_2': os.path.join(output_folder, "stage_2_detection_output"),
        'stage_3': os.path.join(output_folder, "stage_3_face_output"),
        'stage_4': os.path.join(output_folder, "final_output"),
        'colorized': os.path.join(output_folder, "colorized_output")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def copy_input_to_temp_folder(input_file, temp_folder):
    """Copy single input file to a temporary folder for processing"""
    os.makedirs(temp_folder, exist_ok=True)
    filename = os.path.basename(input_file)
    temp_file_path = os.path.join(temp_folder, filename)
    shutil.copy2(input_file, temp_file_path)
    return temp_file_path, temp_folder

def stage_1_overall_restoration(input_file, dirs, opts, gpu_setting):
    """Stage 1: Overall Quality Improvement"""
    logger.info("="*50)
    logger.info("STAGE 1: Overall Quality Restoration")
    logger.info("="*50)
    
    # Create temporary input folder for the single image
    temp_input_dir = os.path.join(dirs['stage_1'], "temp_input")
    _, stage_1_input_dir = copy_input_to_temp_folder(input_file, temp_input_dir)
    
    os.chdir("./Global")
    
    if not opts.with_scratch:
        stage_1_command = (
            f"python test.py --test_mode Full --Quality_restore "
            f"--test_input \"{stage_1_input_dir}\" "
            f"--outputs_dir \"{dirs['stage_1']}\" "
            f"--gpu_ids {gpu_setting}"
        )
        run_cmd(stage_1_command)
    else:
        mask_dir = os.path.join(dirs['stage_1'], "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        
        # Quote paths to handle spaces and special characters
        stage_1_command_1 = (
            f"python detection.py --test_path \"{stage_1_input_dir}\" "
            f"--output_dir \"{mask_dir}\" --input_size full_size "
            f"--GPU {gpu_setting}"
        )

        hr_suffix = " --HR" if opts.HR else ""
        stage_1_command_2 = (
            f"python test.py --Scratch_and_Quality_restore "
            f"--test_input \"{new_input}\" --test_mask \"{new_mask}\" "
            f"--outputs_dir \"{dirs['stage_1']}\" "
            f"--gpu_ids {gpu_setting}{hr_suffix}"
        )

        run_cmd(stage_1_command_1)
        run_cmd(stage_1_command_2)

    # Copy results to final output directory
    stage_1_results = os.path.join(dirs['stage_1'], "restored_image")
    if os.path.exists(stage_1_results):
        for x in os.listdir(stage_1_results):
            if x.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_dir = os.path.join(stage_1_results, x)
                shutil.copy2(img_dir, dirs['stage_4'])
    
    # Cleanup temporary directory
    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    
    cleanup_memory()
    logger.info("Stage 1 completed successfully")

def stage_2_face_detection(dirs, opts, main_environment):
    """Stage 2: Face Detection"""
    logger.info("="*50)
    logger.info("STAGE 2: Face Detection")
    logger.info("="*50)
    
    os.chdir(os.path.join(main_environment, "Face_Detection"))
    stage_2_input_dir = os.path.join(dirs['stage_1'], "restored_image")
    
    if opts.HR:
        stage_2_command = (
            f"python detect_all_dlib_HR.py --url \"{stage_2_input_dir}\" "
            f"--save_url \"{dirs['stage_2']}\""
        )
    else:
        stage_2_command = (
            f"python detect_all_dlib.py --url \"{stage_2_input_dir}\" "
            f"--save_url \"{dirs['stage_2']}\""
        )
    
    run_cmd(stage_2_command)
    cleanup_memory()
    logger.info("Stage 2 completed successfully")

def stage_3_face_enhancement(dirs, opts, gpu_setting, main_environment):
    """Stage 3: Face Enhancement"""
    logger.info("="*50)
    logger.info("STAGE 3: Face Enhancement")
    logger.info("="*50)
    
    os.chdir(os.path.join(main_environment, "Face_Enhancement"))
    stage_3_input_mask = "./"
    stage_3_input_face = dirs['stage_2']
    
    if opts.HR:
        checkpoint_name = 'FaceSR_512'
        stage_3_command = (
            f"python test_face.py --old_face_folder {stage_3_input_face} "
            f"--old_face_label_folder {stage_3_input_mask} "
            f"--tensorboard_log --name {checkpoint_name} "
            f"--gpu_ids {gpu_setting} --load_size 512 --label_nc 18 "
            f"--no_instance --preprocess_mode resize --batchSize 1 "
            f"--results_dir {dirs['stage_3']} --no_parsing_map"
        )
    else:
        stage_3_command = (
            f"python test_face.py --old_face_folder {stage_3_input_face} "
            f"--old_face_label_folder {stage_3_input_mask} "
            f"--tensorboard_log --name {opts.checkpoint_name} "
            f"--gpu_ids {gpu_setting} --load_size 256 --label_nc 18 "
            f"--no_instance --preprocess_mode resize --batchSize 1 "
            f"--results_dir {dirs['stage_3']} --no_parsing_map"
        )
    
    run_cmd(stage_3_command)
    cleanup_memory()
    logger.info("Stage 3 completed successfully")

def stage_4_blending(dirs, opts, main_environment):
    """Stage 4: Face Blending"""
    logger.info("="*50)
    logger.info("STAGE 4: Face Blending")
    logger.info("="*50)
    
    os.chdir(os.path.join(main_environment, "Face_Detection"))
    stage_4_input_image_dir = os.path.join(dirs['stage_1'], "restored_image")
    stage_4_input_face_dir = os.path.join(dirs['stage_3'], "each_img")
    
    if opts.HR:
        stage_4_command = (
            f"python align_warp_back_multiple_dlib_HR.py "
            f"--origin_url {stage_4_input_image_dir} "
            f"--replace_url {stage_4_input_face_dir} "
            f"--save_url {dirs['stage_4']}"
        )
    else:
        stage_4_command = (
            f"python align_warp_back_multiple_dlib.py "
            f"--origin_url {stage_4_input_image_dir} "
            f"--replace_url {stage_4_input_face_dir} "
            f"--save_url {dirs['stage_4']}"
        )
    
    run_cmd(stage_4_command)
    cleanup_memory()
    logger.info("Stage 4 completed successfully")

def stage_5_colorization(input_file, dirs, skip_colorization=False):
    """Stage 5: Image Colorization"""
    if skip_colorization:
        logger.info("Skipping colorization as requested")
        return None
    
    logger.info("="*50)
    logger.info("STAGE 5: Image Colorization")
    logger.info("="*50)
    
    try:
        # Find the final processed image
        final_image_path = None
        
        # Check final_output directory first
        if os.path.exists(dirs['stage_4']):
            for file in os.listdir(dirs['stage_4']):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    final_image_path = os.path.join(dirs['stage_4'], file)
                    break
        
        # If not found, check stage_1 output
        if not final_image_path:
            stage_1_restored = os.path.join(dirs['stage_1'], "restored_image")
            if os.path.exists(stage_1_restored):
                for file in os.listdir(stage_1_restored):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        final_image_path = os.path.join(stage_1_restored, file)
                        break
        
        if not final_image_path:
            logger.warning("No processed image found for colorization. Using original image.")
            final_image_path = input_file
        
        # Colorize the image
        filename = os.path.basename(final_image_path)
        colorized_output_path = os.path.join(dirs['colorized'], f"colorized_{filename}")
        
        colorized_path = colorize_image(final_image_path, colorized_output_path)
        
        cleanup_memory()
        logger.info(f"Colorization completed. Output saved to: {colorized_path}")
        return colorized_path
        
    except Exception as e:
        logger.error(f"Error during colorization: {str(e)}")
        logger.info("Colorization failed, continuing without it")
        return None

def find_final_result(input_file, dirs):
    """Find and return the path to the final processed image"""
    # Priority order: colorized -> final_output -> stage_1_restored -> original
    search_locations = [
        (dirs['colorized'], "colorized image"),
        (dirs['stage_4'], "final processed image"),
        (os.path.join(dirs['stage_1'], "restored_image"), "stage 1 restored image"),
    ]
    
    for location, description in search_locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    result_path = os.path.join(location, file)
                    logger.info(f"Final result: {description} at {result_path}")
                    return result_path, description
    
    # If nothing found, copy original to final directory
    logger.warning("No processed images found. Copying original image to final directory.")
    original_filename = os.path.basename(input_file)
    fallback_path = os.path.join(dirs['stage_4'], f"original_{original_filename}")
    shutil.copy2(input_file, fallback_path)
    return fallback_path, "original image (processing may have failed)"

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced Photo Restoration with CPU optimization and colorization")
    parser.add_argument("--input_file", type=str, required=True, help="Input image file path")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for restored images")
    parser.add_argument("--GPU", type=str, default="-1", help="GPU IDs (-1 for CPU mode)")
    parser.add_argument("--checkpoint_name", type=str, default="Setting_9_epoch_100", help="Checkpoint name")
    parser.add_argument("--with_scratch", action="store_true", help="Enable scratch detection and removal")
    parser.add_argument("--HR", action='store_true', help="Enable high resolution processing")
    parser.add_argument("--skip_colorization", action='store_true', help="Skip the colorization stage")
    
    opts = parser.parse_args()

    # Validate input file
    if not os.path.exists(opts.input_file):
        logger.error(f"Input file does not exist: {opts.input_file}")
        sys.exit(1)

    # Setup GPU/CPU mode
    gpu_setting = opts.GPU
    if gpu_setting == "-1":
        logger.info("Running in CPU mode")
        # Set environment variables for CPU optimization
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count()))
    else:
        logger.info(f"Running in GPU mode with GPU IDs: {gpu_setting}")

    # Resolve paths
    opts.input_file = os.path.abspath(opts.input_file)
    opts.output_folder = os.path.abspath(opts.output_folder)
    main_environment = os.getcwd()

    # Setup directories
    dirs = setup_directories(opts.output_folder)

    logger.info("="*60)
    logger.info("ENHANCED PHOTO RESTORATION PIPELINE")
    logger.info("="*60)
    logger.info(f"Input file: {opts.input_file}")
    logger.info(f"Output folder: {opts.output_folder}")
    logger.info(f"GPU setting: {gpu_setting}")
    logger.info(f"High resolution: {opts.HR}")
    logger.info(f"Scratch detection: {opts.with_scratch}")
    logger.info(f"Skip colorization: {opts.skip_colorization}")

    try:
        # Stage 1: Overall Quality Improvement
        stage_1_overall_restoration(opts.input_file, dirs, opts, gpu_setting)
        os.chdir(main_environment)

        # Stage 2: Face Detection
        stage_2_face_detection(dirs, opts, main_environment)
        os.chdir(main_environment)

        # Stage 3: Face Enhancement
        stage_3_face_enhancement(dirs, opts, gpu_setting, main_environment)
        os.chdir(main_environment)

        # Stage 4: Face Blending
        stage_4_blending(dirs, opts, main_environment)
        os.chdir(main_environment)

        # Stage 5: Colorization
        colorized_path = stage_5_colorization(opts.input_file, dirs, opts.skip_colorization)
        os.chdir(main_environment)

        # Find and report final result
        final_result_path, description = find_final_result(opts.input_file, dirs)
        
        logger.info("="*60)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Final result: {description}")
        logger.info(f"Output path: {final_result_path}")
        logger.info(f"All outputs saved in: {opts.output_folder}")
        
        # Final memory cleanup
        cleanup_memory()

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error("Check the logs above for more details")
        os.chdir(main_environment)
        sys.exit(1)