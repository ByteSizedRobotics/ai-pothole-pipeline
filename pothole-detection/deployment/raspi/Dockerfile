# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# COPY yolov5s.pt /app/models/yolo5s.pt
# COPY best_20250202.pt /app/models/best_20250202.pt

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1

WORKDIR /app
COPY . /app
RUN mkdir /app/uploads

# Declare /app/saved_images as a volume (now using a bind mount)
# VOLUME /app/saved_images

# Set environment variable for save path
ENV SAVE_PATH="/app/saved_images"

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "app.py"]
