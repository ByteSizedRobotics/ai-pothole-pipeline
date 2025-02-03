# ai-pothole-models

## Classification Model
To run docker container

- login to docker in terminal (docker login)
- pull the image (docker pull ngawargy/pothole-classifier-app:latest)
- run (docker run -p 5000:5000 ngawargy/pothole-classifier-app:latest)
- go to http://localhost:5000 to see the app


To deploy flask app.py on local web server without Docker (might need to install pip dependencies 1st):

- cd to deployment\local-app
- python app.py
- go to the specified web address (ex: http://127.0.0.1:5000/)

To run it locally (not on web server but just as python script)
- cd to deployment\testing
- python local-model-testing.py


## Setup Virtual Env and install dependencies:

- python -m venv venv

2) To activate the venv:

- Set-ExecutionPolicy Unrestricted -Scope Process
- .\venv\Scripts\Activate
- Run the local-app.py files: python name_file.py

3) Install required packages:

- cd to pothole-detection\deployment\local-app
- pip install -r requirements.txt


