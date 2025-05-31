from flask import Flask, send_from_directory
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    CORS(app)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
    app.config['RESULTS_FOLDER'] = os.path.join('app', 'static', 'results')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure upload and results directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Log directory creation
    app.logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Created results folder: {app.config['RESULTS_FOLDER']}")
    
    # Register blueprints
    from .routes import main
    app.register_blueprint(main)
    
    # Add route to serve static files
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory(app.static_folder, filename)
    
    return app 