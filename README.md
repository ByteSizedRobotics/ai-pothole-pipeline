# ai-pothole-models

## Local Classification Model
#### 1) Deploy flask local_app.py on local web server without Docker 
(need to install dependencies 1st.. see Setup Virtual Env Steps)

- cd to deployment\local-app
- python local_app.py
- go to the specified web address (ex: http://127.0.0.1:5000/)

#### 2) Running python script locally (not on web server but just as python script)
- cd to deployment\testing
- python local-model-testing.py

## Docker Classification Model
Only the live inference with video is done in the container due to permission/access complexities for providing access to host files/folder within the container. For now, a bind mount is used as the location to share access between the host and the container for the detected/saved images. 
#### 1) Building and Pushing Docker Container
- docker buildx build --platform linux/arm64 -t [dockerUSERNAME]/pothole-classifier-app-arm64 .
- docker push [dockerUSERNAME]/pothole-classifier-app-arm64

#### 2) Deploy app.py on Docker Container on Raspberry Pi 
- docker pull --platform linux/arm64 ngawargy/pothole-classifier-app-arm64
- docker run -d --name flask-pothole --device=/dev/video0 -p 5000:5000 -v [PATH ON HOST TO SAVE IMAGES]:/app/saved_images your_image ngawargy/pothole-classifier-app-arm64

## Setup Virtual Env and install dependencies

- python -m venv venv

2) To activate the venv:

- Set-ExecutionPolicy Unrestricted -Scope Process
- .\venv\Scripts\Activate
- Run the local-app.py files: python name_file.py

3) Install required packages:

- cd to pothole-detection\deployment\local-app
- pip install -r requirements.txt


