# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libavresample-dev \
    libopus-dev \
    libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support first (this is the largest dependency)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the entire application
COPY . .

# Create directories for AI models if they don't exist
RUN mkdir -p aimodels/DepthAnythingV2/checkpoints \
    && mkdir -p aimodels/DeepLabV3Plus/checkpoints \
    && mkdir -p aimodels/yolov5

# Download YOLOv5 dependencies (this will be cached in the ultralytics hub)
RUN python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)"

# Set environment variables for the application
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose any ports if needed (adjust as necessary)
EXPOSE 5000 8765

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('PyTorch available:', torch.cuda.is_available())" || exit 1

# Run the application
CMD ["python", "main.py"]