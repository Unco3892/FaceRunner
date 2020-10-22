import cv2
import sys
import dlib
from math import hypot

""" Load the classifier """
face_cascade = cv2.CascadeClassifier(
    'resources/haarcascade_frontalface_default.xml')

emoji_image = cv2.imread('graphics/anger.png')

""" Loading the detector from the dlib library """
detector = dlib.get_frontal_face_detector()

""" To get the landmarks of the face"""
predictor = dlib.shape_predictor(
    "resources/shape_predictor_68_face_landmarks.dat")

""" Define the functions """


def randomizer():
    print('Test randomizer')


def smiley():
    print('Test smiley')


"""We can cut down from our code in the webcam face detector and emoji 
replacer by getting it to return (cap) in some way"""


# def webcam_checker():
#     print('Let\'s Start the Webcam!')
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise IOError("Cannot open webcam!")


def webcam_face_detector():
    print('Let\'s Start the Webcam!')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")
    while True:
        """Analyze the images the webcam image at 1ms frame rate"""
        _, frame = cap.read()
        """Starting the webcam and converting  the image to a grayscale image to 
        enable face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """ detecting the faces from the dlib library """
        faces = detector(gray)
        """ extracting the coordinate of the faces """
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            """ draw the rectangle on the frame """
            """ detect the landmark point (face points) """
            landmarks = predictor(gray, face)
            """ Showing the coordinates of the face based on the 68 points 
            in case you have quesitons on this part, please see 
            "resources/face-mark-points.png" """
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                """ drawing those coordinates """
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        """ Show the frame """
        cv2.imshow("Face Detector 2020", frame)
        """ Displaying a frame for 1 ms, after which display will 
        be automatically closed"""
        key = cv2.waitKey(1)
        """Break the loop if the 27th key, meaning the escape key was pressed"""
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def webcam_emoji_replacer():
    print('Let\'s Start the Webcam!')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")
    while True:
        """ Analyze the images the webcam image at 1ms frame rate """
        _, frame = cap.read()
        """Start the webcam convert the image to a grayscale image to enable 
        face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """ detecting the faces from the dlib library """
        faces = detector(gray)
        """ We are interested in three points of the face coordinates, 
        namely 19, 25 and 30 --> To understand better refer again to the 
        face-mark-points"""
        for face in faces:
            landmarks = predictor(gray, face)
            left_forehead = (landmarks.part(19).x, landmarks.part(19).y)  #
            right_forehead = (landmarks.part(25).x, landmarks.part(25).y)
            center_face = (landmarks.part(30).x, landmarks.part(30).y)
            """Creating an adjustable emoji based on the width of the face as
             well as the height, the ratio of h to w in the emoji image is 
             1. For the diameter we take index 0 as the first element and index 
             1 as the second element"""
            diameter = int(hypot(left_forehead[0] - right_forehead[0],
                                 left_forehead[1] - right_forehead[1]) * 2.3)
            """ Getting the top left of the face """
            top_left = (int(center_face[0] - diameter / 2),
                        int(center_face[1] - diameter / 2))

            """ Getting the bottom right of the face """
            bottom_right = (int(center_face[0] + diameter / 2),
                            int(center_face[1] + diameter / 2))

            """ Resizing the imported emoji """
            emoji_face = cv2.resize(emoji_image, (diameter, diameter))
            emoji_face_gray = cv2.cvtColor(emoji_face, cv2.COLOR_BGR2GRAY)

            """ Using the mask overlay """
            _, face_mask = cv2.threshold(emoji_face_gray, 25, 255,
                                         cv2.THRESH_BINARY_INV)

            """ Using the mask overlay """
            face_area = frame[top_left[1]: top_left[1] + diameter,
                        top_left[0]: top_left[0] + diameter]

            """ Taking the face out and applying the mask """
            face_area_no_face = cv2.bitwise_and(face_area, face_area,
                                                mask=face_mask)

            """ Adding the two images together """
            final_face = cv2.add(face_area_no_face, emoji_face)

            """ We set the array equal to the final_face  """
            frame[top_left[1]: top_left[1] + diameter,
            top_left[0]: top_left[0] + diameter] = final_face

        """ Showing the final result """
        cv2.imshow("Emoji Replacer 2020", frame)
        key = cv2.waitKey(1)
        """ Break the loop if the 27th key, meaning the escape key was pressed """
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


""" This function requests an input image from the user and detects faces in 
the image. Then, it provides the user with the option to either save the 
image or close the image."""


def picture():
    # Print instructions for the user
    print('\nInstructions:')
    print(
        'Before starting, place the image that you would like to use in the same'
        ' folder as this program.')
    print('While your image is displayed, press:')
    print('1) To save your image.')
    print('0) To close the image.\n')
    """ Draw on top of the images """

    while True:
        """ Take in the filename from the user and load the image """
        image_name = input(
            'Enter the name + extension of your image (e.g. Test.jpg)')
        img = cv2.imread(image_name)
        try:
            """ Convert the image to a grayscale image to enable face detection """
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            break
        # Catch a conversion error that occurs when the  filename/extension
        # is incorrect
        except cv2.error:
            print('Image name or extension are incorrect. Please try again.')

    """ Detect faces in the image """
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    """ Place rectangles around the faces in the image """
    # TODO: Replace with emoji code
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    """ Show the end result to the user """
    cv2.imshow('img', img)
    """ Detect valid key presses from the user and perform appropriate actions 
 accordingly """
    key = cv2.waitKey()
    if key == 0:
        cv2.destroyAllWindows()
    elif key == 1:
        cv2.imwrite('Output.jpg', img)


""" This function prints the main menu."""


def menu():
    print('\nWelcome to our face detection & replacement tool!')
    print(
        'This program allows you to either select an existing image or stream your webcam.')
    print(
        'Then, it will detect the faces in the image and show in real time '
        'and also if you would like to, it can replace your face with a random '
        'emoji.')
    print('Finally, you will be able to save the end result.\n')
    print('What would you like to do:')
    print('1) Use an existing image.')
    print('2) Use your webcam to detect faces.')
    print('3) Use your webcam to replace your face with a random emoji.')
    print('0) Exit the program. \n')


"""""""""""""""""""""""""""
Main code
"""""""""""""""""""""""""""

"""Call the main menu."""
menu()
""" Take the users input, according to the options provided in the main menu."""
""" If the input is invalid, prompt the user to try again."""
while True:
    try:
        choice = int(input('Enter your choice:'))
        if choice == 1:
            picture()
            menu()
        elif choice == 2:
            webcam_face_detector()
            menu()
        elif choice == 3:
            webcam_emoji_replacer()
            menu()
        elif choice == 0:
            sys.exit()
        else:
            print('That was not a valid number. Try again.')
            continue
    except ValueError:
        print("That was not a number. Try again.")
