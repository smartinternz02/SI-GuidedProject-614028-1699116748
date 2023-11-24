"""
Constants used in the project
"""
import os

IMAGE_SIZE = (224, 224)
MAX_LENGTH = 34
UPLOAD_FOLDER = os.path.join("static", "uploads")

# check if the folder exists if not create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(["png", "jpg"])
