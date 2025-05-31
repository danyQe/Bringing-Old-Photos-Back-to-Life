from app import create_app
import os
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler()
    ]
)

app = create_app()

if __name__ == '__main__':
    # Log system information
    logging.info(f"Running on {platform.system()} {platform.release()}")
    logging.info(f"Python version: {platform.python_version()}")
    
    # Ensure the required directories exist
    os.makedirs('app/static/uploads', exist_ok=True)
    os.makedirs('app/static/results', exist_ok=True)
    
    # Log directory creation
    app.logger.info("Created upload and results directories")
    
    # Run the Flask application with increased timeout
    app.run(host='0.0.0.0', port=5000, threaded=True) 