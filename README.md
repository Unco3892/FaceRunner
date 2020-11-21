# **FaceRunner**: Face detection and overlay
Presented to [Dr. Mario Silic] (https://www.alexandria.unisg.ch/persons/6160) by Ilia Azizi and Emile Evers for the [course in programming at the University of St. Gallen 2020](https://codingxcamp.com/) 
## Introduction

This project aims to detect faces and some additional overlays from photos or live video streams broadcasted through the webcam.

## Demo Video
To see a demo video, click on the photo [![Face Runner Demo](https://github.com/eversemile/codingcamp/blob/main/utilities/video-thumb.jpg?raw=true)](https://rec.unil.ch/lti/v125f70a5e739oem2u8n/)
## Dependencies

* Python 3.8.6, [OpenCV 4](https://pypi.org/project/opencv-python/), [dlib 19](https://pypi.org/project/dlib/), [pyfiglet 0.8](https://pypi.org/project/pyfiglet/)
* To install the required packages, run `pip install -r requirements.txt`

**Note**: If you have recieve an error for the `dlib` library, dlib is an c++ libary with python bindings. It needs to be builded first. To do so you must run:
- run `pip install cmake` , if that didn't work you can also install it manually from [here](https://cmake.org/download/)
- Install Visual Studio build tools from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15#).
- In Visual Studio 2017 or other versions, go to the Individual Components tab, Visual C++ Tools for Cmake, and check the checkbox under the "Compilers, build tools and runtimes" section.
-  If all steps were unsuccessful, you can also follow the instructions by various people [here](https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10).

## Usage
To run the program type `python Main.py`

## References and models
The frontalface used for this software come from [opencv's haarcascades](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml). Additionally, the landmarks for the face were trained by [Davis King](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) using the dlib library. All the supplementary files can be found in the [utilities](https://github.com/eversemile/codingcamp/tree/main/utilities) folder.

## Room for imporvements
The current overlays do rotate with the face which could be improved using more advanced landmarkings and techniques.
