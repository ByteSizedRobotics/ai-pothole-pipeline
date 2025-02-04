# Scripts to deploy the Flask app locally using Docker
# NOTE: Docker Desktop needs to be running

#!/bin/bash

IMAGE_NAME="ngawargy/pothole-classifier-app"

# TODO: you need to specifiy the directory where you want to share files between the host and the container
# this directory will be used to store the uploaded images and so container has access to it
CONTAINER_NAME="flask-pothole"

echo "Pulling the latest Docker image"
docker pull $IMAGE_NAME

# stop and remove any existing container with the same name
if [ $(docker ps -aq -f name=$CONTAINER_NAME) ]; then
    echo "Stopping and removing existing container"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

echo "Running the Docker container" # allows camera access and mounts the shared directory
# should work on LINUX, issues on Windows accessing the camera
docker run -d \
    --name $CONTAINER_NAME \
    --device=/dev/video0 \
    -p 5000:5000 \
    $IMAGE_NAME

echo "Deployment script complete. Access the app at http://localhost:5000"
