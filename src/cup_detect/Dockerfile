# Use Python 3.10 slim image
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install system dependencies: build tools + CMake + libs needed for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    **build-essential** \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Default command
CMD ["python", "cup_detect.py"]


# sudo docker run -it --device /dev/video2:/dev/video2 --gpus '"device=0"' -p 6000:6000 507c10e0d63f /bin/bash
