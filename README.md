# Face detection and overlay
## Introduction

This project aims to detect faces and some additional overlays from photos or live video streams broadcasted through the webcam.

## Dependencies

* Python 3, [OpenCV 4](https://pypi.org/project/opencv-python/), [dlib 19](https://pypi.org/project/dlib/), [pyfiglet 0.8](https://pypi.org/project/pyfiglet/), 
* To install the required packages, run `pip install -r requirements.txt`.

## Usage
To run the program `python Main.py`

## References and models
The frontalface used for this software come from [opencv's haarcascades](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml). Additionally, the landmarks for the face were trained by [Davis King](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) using the dlib library. All the supplementary files can be found in the [utilities](https://github.com/eversemile/codingcamp/tree/main/utilities) folder.

## Room for imporvements
The current overlays do rotate with the face which could be improved using more advanced landmarkings and techniques.
