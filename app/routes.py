from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app
import os
from werkzeug.utils import secure_filename
import subprocess
import uuid
import shutil

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            new_filename = f"{unique_id}.{file_ext}"
            
            # Save file
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], new_filename)
            file.save(file_path)
            
            # Log file save
            current_app.logger.info(f"File saved to: {file_path}")
            
            # Get processing options
            options = {
                'with_scratch': request.form.get('with_scratch', 'false').lower() == 'true',
                'high_resolution': request.form.get('high_resolution', 'false').lower() == 'true',
                'detect_scratches': request.form.get('detect_scratches', 'false').lower() == 'true',
                'colorize': request.form.get('colorize', 'true').lower() == 'true'
            }
            
            # Log processing options
            current_app.logger.info(f"Processing options: {options}")
            
            # Process the image
            try:
                result_path = process_image(file_path, options)
                # Convert the absolute path to a URL path
                relative_path = os.path.relpath(result_path, current_app.config['RESULTS_FOLDER'])
                # Ensure the path uses forward slashes for URLs
                url_path = f"/static/results/{relative_path.replace(os.sep, '/')}"
                current_app.logger.info(f"Processing successful, result path: {url_path}")
                return jsonify({
                    'success': True,
                    'result_path': url_path
                })
            except Exception as e:
                # Log the error for debugging
                current_app.logger.error(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Failed to process image: {str(e)}'}), 500
        except Exception as e:
            current_app.logger.error(f"Error handling upload: {str(e)}")
            return jsonify({'error': f'Error handling upload: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_image(file_path, options):
    # Create output directory for this processing
    output_dir = os.path.join(current_app.config['RESULTS_FOLDER'], os.path.basename(file_path).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Create stage directories
    stage_1_dir = os.path.join(output_dir, "stage_1_restore_output")
    stage_2_dir = os.path.join(output_dir, "stage_2_detection_output")
    stage_3_dir = os.path.join(output_dir, "stage_3_face_output")
    final_dir = os.path.join(output_dir, "final_output")
    colorized_dir = os.path.join(output_dir, "colorized_output")
    
    os.makedirs(stage_1_dir, exist_ok=True)
    os.makedirs(stage_2_dir, exist_ok=True)
    os.makedirs(stage_3_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(colorized_dir, exist_ok=True)
    
    # Build command based on options - now using --input_file for direct file processing
    cmd = ['python', 'run.py',
           '--input_file', file_path,
           '--output_folder', output_dir,
           '--GPU', '-1']  # Use CPU mode (-1) instead of GPU (0)
    
    if options['with_scratch']:
        cmd.append('--with_scratch')
    
    if options['high_resolution']:
        cmd.append('--HR')
    
    if not options['colorize']:
        cmd.append('--skip_colorization')
    
    # Log the command for debugging
    current_app.logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command with a timeout
    try:
        # Use threading for timeout instead of signal (which doesn't work on Windows)
        import threading
        import time
        
        # Create a flag to track if the process has completed
        process_completed = False
        process_result = None
        process_error = None
        
        def run_process():
            nonlocal process_completed, process_result, process_error
            try:
                # Capture output for debugging
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                process_result = result
                process_completed = True
            except Exception as e:
                process_error = e
                process_completed = True
        
        # Start the process in a separate thread
        process_thread = threading.Thread(target=run_process)
        process_thread.daemon = True
        process_thread.start()
        
        # Wait for the process to complete or timeout
        timeout_seconds = 3600  # 60 minutes (1 hour)
        start_time = time.time()
        
        while not process_completed:
            if time.time() - start_time > timeout_seconds:
                current_app.logger.error("Image processing timed out after 60 minutes")
                raise TimeoutError("Image processing timed out after 60 minutes")
            time.sleep(1)
        
        # Check if there was an error
        if process_error:
            raise process_error
        
        # Get the result
        result = process_result
        current_app.logger.info(f"Command output: {result.stdout}")
        
        if result.stderr:
            current_app.logger.warning(f"Command stderr: {result.stderr}")
        
        # Check if the output directories exist and log their contents
        current_app.logger.info(f"Checking output directories:")
        
        # First check for colorized output if colorization was enabled
        if options['colorize'] and os.path.exists(colorized_dir):
            current_app.logger.info(f"Colorized dir exists: {os.path.exists(colorized_dir)}")
            current_app.logger.info(f"Colorized dir contents: {os.listdir(colorized_dir) if os.path.exists(colorized_dir) else 'N/A'}")
            
            # Try to find a colorized image
            if os.path.exists(colorized_dir) and os.listdir(colorized_dir):
                found_file = None
                for file in os.listdir(colorized_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        found_file = os.path.join(colorized_dir, file)
                        break
                
                if found_file:
                    return found_file
        
        # If no colorized output or colorization disabled, check final output
        current_app.logger.info(f"Final dir exists: {os.path.exists(final_dir)}")
        if os.path.exists(final_dir):
            current_app.logger.info(f"Final dir contents: {os.listdir(final_dir)}")
        
        current_app.logger.info(f"Stage 1 dir exists: {os.path.exists(stage_1_dir)}")
        if os.path.exists(stage_1_dir):
            current_app.logger.info(f"Stage 1 dir contents: {os.listdir(stage_1_dir)}")
            restored_image_dir = os.path.join(stage_1_dir, "restored_image")
            current_app.logger.info(f"Restored image dir exists: {os.path.exists(restored_image_dir)}")
            if os.path.exists(restored_image_dir):
                current_app.logger.info(f"Restored image dir contents: {os.listdir(restored_image_dir)}")
        
        # Return path to the final result
        result_path = os.path.join(final_dir, os.path.basename(file_path))
        if not os.path.exists(result_path):
            # If the file doesn't exist in final_output, check stage_1_restore_output
            stage_1_result = os.path.join(stage_1_dir, "restored_image", os.path.basename(file_path))
            if os.path.exists(stage_1_result):
                result_path = stage_1_result
            else:
                # Check if the file exists with a different name in the directories
                found_file = None
                
                # Check in final_dir
                if os.path.exists(final_dir):
                    for file in os.listdir(final_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            found_file = os.path.join(final_dir, file)
                            break
                
                # Check in stage_1_dir/restored_image
                if not found_file and os.path.exists(stage_1_dir):
                    restored_image_dir = os.path.join(stage_1_dir, "restored_image")
                    if os.path.exists(restored_image_dir):
                        for file in os.listdir(restored_image_dir):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                found_file = os.path.join(restored_image_dir, file)
                                break
                
                if found_file:
                    result_path = found_file
                else:
                    # If nothing was found, return the original uploaded image as a last resort
                    current_app.logger.warning("No processed image found in any output directory. Returning original image.")
                    
                    # Copy the original file to the results directory so it's accessible via URL
                    original_result_path = os.path.join(final_dir, "original_" + os.path.basename(file_path))
                    try:
                        shutil.copy(file_path, original_result_path)
                        current_app.logger.info(f"Copied original image to: {original_result_path}")
                        return original_result_path
                    except Exception as e:
                        current_app.logger.error(f"Error copying original image: {str(e)}")
                        # Return the original path if we can't copy
                        return file_path
                
        return result_path
    except TimeoutError as e:
        current_app.logger.error(f"Image processing timed out: {str(e)}")
        raise
    except subprocess.CalledProcessError as e:
        current_app.logger.error(f"Error running image processing: {str(e)}")
        current_app.logger.error(f"Command stderr: {e.stderr}")
        raise
    except Exception as e:
        current_app.logger.error(f"Unexpected error during image processing: {str(e)}")
        raise

@main.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(current_app.config['RESULTS_FOLDER'], filename) 

