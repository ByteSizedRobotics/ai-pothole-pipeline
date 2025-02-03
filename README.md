# ai-pothole-models

## Classification Model
To deploy flask app.py on local web server (might need to install pip dependencies 1st):

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

- cd to pothole-detection\deployment
- pip install -r requirements.txt


