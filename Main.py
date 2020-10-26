import cv2
import sys
import dlib
from math import hypot

""" Load the classifier """
face_cascade = cv2.CascadeClassifier(
    'utilities/haarcascade_frontalface_default.xml')

""" Importing the emoji """
emoji_image = cv2.imread('graphics/surprise.png')

""" Importing the mask face """
mask_image = cv2.imread('utilities/custom_mask.png')

""" Loading the detector from the dlib library """
detector = dlib.get_frontal_face_detector()

""" To get the landmarks of the face"""
predictor = dlib.shape_predictor(
    "utilities/shape_predictor_68_face_landmarks.dat")

"""We can cut down from our code in the webcam face detector and emoji 
replacer by getting it to return (cap) in some way"""


# def webcam_checker():
#     print('Let\'s Start the Webcam!')
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise IOError("Cannot open webcam!")

def emoji_overlay(gray, input):
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

        """ Resizing the imported emoji """
        emoji_face = cv2.resize(emoji_image, (diameter, diameter))
        emoji_face_gray = cv2.cvtColor(emoji_face, cv2.COLOR_BGR2GRAY)

        """ Using the mask overlay """
        _, face_mask = cv2.threshold(emoji_face_gray, 25, 255,
                                        cv2.THRESH_BINARY_INV)

        """ Using the mask overlay """
        face_area = input[top_left[1]: top_left[1] + diameter,
                    top_left[0]: top_left[0] + diameter]

        """ Taking the face out and applying the mask """
        face_area_no_face = cv2.bitwise_and(face_area, face_area,
                                            mask=face_mask)

        """ Adding the two images together """
        final_face = cv2.add(face_area_no_face, emoji_face)

        """ We set the array equal to the final_face  """
        input[top_left[1]: top_left[1] + diameter,
        top_left[0]: top_left[0] + diameter] = final_face

    return input


def webcam_face_detector():
    print('Let\'s Start the Webcam!')
    print('Note: Press \'Escape\' to exit the webcam.')
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
            "utilities/face-mark-points.png" """
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
    print('Note: Press \'Escape\' to exit the webcam.')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")
    while True:
        try:
            """ Analyze the images the webcam image at 1ms frame rate """
            _, frame = cap.read()
            """Start the webcam convert the image to a grayscale image to enable 
            face detection"""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            """Detect faces in the current frame and overlay them with emoji's"""
            frame = emoji_overlay(gray, frame)

            """ Showing the final result """
            cv2.imshow("Emoji Replacer 2020", frame)
            key = cv2.waitKey(1)
            """ Break the loop if the 27th key, meaning the escape key was pressed """
            if key == 27:
                break
        except cv2.error:
           pass
    cap.release()
    cv2.destroyAllWindows()


def webcam_mask():
    print('Let\'s Start the Webcam!')
    print('Note: Press \'Escape\' to exit the webcam.')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")
    while True:
        try:
            """ Analyze the images the webcam image at 1ms frame rate """
            _, frame = cap.read()
            """Start the webcam convert the image to a grayscale image to enable 
            face detection"""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            """ detecting the faces from the dlib library """
            faces = detector(gray)
            """ Once again we use the face coordinates but a little differently 
            this time"""
            for face in faces:
                landmarks = predictor(gray, face)
                left_ear = (landmarks.part(2).x, landmarks.part(2).y)  #
                right_ear = (landmarks.part(14).x, landmarks.part(14).y)
                center_face = (landmarks.part(33).x, landmarks.part(33).y)
                mouth = (landmarks.part(66).x, landmarks.part(66).y)
                """Creating an adjustable mask based on the width of the face as
                well as the height, the height of the mask is 1085 and the width is
                1627 giving us a ratio of 0.66"""
                face_width = int(hypot(left_ear[0] - right_ear[0],
                                    left_ear[1] - right_ear[1]) * 1.6)
                chin_to_nose = int(face_width * 0.66)

                """ Getting the top left of the face """
                top_left = (int(mouth[0] - face_width / 2),
                            int(mouth[1] - chin_to_nose / 2))

                """ Resizing the imported mask """
                mask_face = cv2.resize(mask_image, (face_width, chin_to_nose))
                mask_face_gray = cv2.cvtColor(mask_face, cv2.COLOR_BGR2GRAY)

                """ Using the mask overlay """
                # _, face_mask = cv2.threshold(mask_face_gray, 130, 255,
                #                              cv2.THRESH_BINARY_INV)
                _, face_mask = cv2.threshold(mask_face_gray, 0, 255,
                                            cv2.THRESH_BINARY_INV)

                """ Using the mask overlay """
                face_area = frame[top_left[1]: top_left[1] + chin_to_nose,
                            top_left[0]: top_left[0] + face_width]

                """ Taking the face out and applying the mask """
                face_area_no_face = cv2.bitwise_and(face_area, face_area,
                                                    mask=face_mask)

                """ Adding the two images together """
                final_face = cv2.add(face_area_no_face, mask_face)

                """ We set the array equal to the final_face  """
                frame[top_left[1]: top_left[1] + chin_to_nose,
                top_left[0]: top_left[0] + face_width] = final_face

            """ Showing the final result """
            cv2.imshow("PROTECT YOURSELF 2020!", frame)
            key = cv2.waitKey(1)
            """ Break the loop if the 27th key, meaning the escape key was pressed """
            if key == 27:
                break
        except cv2.error:
            pass     
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
    print('Note: Press \'Escape\' to exit the picture.')

    """ Draw on top of the images """
    while True:
        """ Take in the filename from the user and load the image """
        image_name = input(
            'Enter the name + extension of your image (e.g. Test.jpg): ')
        img = cv2.imread(image_name)
        try:
            """ Convert the image to a grayscale image to enable face detection """
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            break
        # Catch a conversion error that occurs when the  filename/extension
        # is incorrect
        except cv2.error:
            print('Image name or extension are incorrect. Please try again.')

    """Detect faces in the image and replace them with emoji's"""
    img = emoji_overlay(gray, img)
    
    """ Show the result at the end ."""
    cv2.imshow('Faces Detected!', img)
    """ Detect valid key presses from the user and perform appropriate actions accordingly """
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()


""" This function is for some nice printing capabilities for subheaders """
def print_structure(str, n):
    hyphens = "-" * int((n - len(str)) / 2)
    str_p = hyphens + " " + str + " " + hyphens
    hyphens_bar = "-" * len(str_p)
    print(hyphens_bar)
    print(str_p)
    print(hyphens_bar)


""" The function displays the header of the project"""
def display_header():
    with open("utilities/header.txt", "r") as file:
        for line in file:
            cc = 1
            print(line, end="")


""" This function prints the main menu."""
def menu():
    display_header()
    print_structure('Welcome to our face detection & replacement tool!', 60)
    # print()
    print(
        'This program allows you to either select an existing image or '
        '\nstream your webcam.')
    print(
        'Then, it will detect the faces in the image and show in real\ntime '
        'and also if you would like to, it can replace your face\nwith an '
        'expression of our choice '
        'emoji.')
    print('What would you like to do:')
    print('1) Use an existing image to detect faces.')
    print('2) Use your webcam to detect faces.')
    print('3) Use your webcam to be shocked.')
    print('4) Use your webcam to be protected immediately!')
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
        elif choice == 4:
            webcam_mask()
            menu()
        elif choice == 0:
            sys.exit()
        else:
            print('That was not a valid number. Try again.')
            continue
    except ValueError:
        print("That was not a number. Try again.")
