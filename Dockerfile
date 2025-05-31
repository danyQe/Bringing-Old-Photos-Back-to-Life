FROM ubuntu:20.04

# Install dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -yq \
    git \
    bzip2 \
    wget \
    unzip \
    python3-pip \
    python3-dev \
    cmake \
    libgl1-mesa-dev \
    python-is-python3 \
    libgtk2.0-dev \
    libglib2.0-0

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Set up the synchronized batch norm
RUN cd Face_Enhancement/models/networks/ && \
    git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch && \
    cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm . && \
    cd ../../../

RUN cd Global/detection_models && \
    git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch && \
    cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm . && \
    cd ../../

# Download face landmark detection model
RUN cd Face_Detection/ && \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    cd ../

# Download pretrained models
RUN cd Face_Enhancement/ && \
    wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip && \
    unzip checkpoints.zip && \
    cd ../ && \
    cd Global/ && \
    wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip && \
    unzip checkpoints.zip && \
    rm -f checkpoints.zip && \
    cd ../

# Download colorization model files
RUN cd Global/models/ && \
    wget https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt && \
    wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel && \
    wget https://github.com/richzhang/colorization/raw/caffe/resources/pts_in_hull.npy && \
    cd ../../

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir numpy dlib 

# Install OpenCV with binary package to ensure it works properly
RUN pip3 install --no-cache-dir --only-binary=:all: opencv-python

RUN pip3 install --no-cache-dir -r requirements.txt

# Install Flask and related dependencies for web interface
RUN pip3 install --no-cache-dir flask flask-cors gunicorn Werkzeug gevent

# Create necessary directories for the web app
RUN mkdir -p app/static/uploads app/static/results

# Expose port for web interface
EXPOSE 5000

# Create a directory for test images
RUN mkdir -p test_images/old

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "web" ]; then\n\
  # Run Flask web application\n\
  python run_flask.py\n\
elif [ "$1" = "cli" ]; then\n\
  # Run CLI version with provided arguments\n\
  shift\n\
  python run.py "$@"\n\
else\n\
  # Default: run web app\n\
  python run_flask.py\n\
fi' > /app/docker-entrypoint.sh && \
chmod +x /app/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (will run web interface by default)
CMD ["web"]
