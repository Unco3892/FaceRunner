import cv2
import sys

def randomizer():
    print('Test randomizer')

def smiley():
    print('Test smiley')

def webcam():
    print('Test webcam')

# This function requests an input image from the user and detects faces in the image. 
# Then, it provides the user with the option to either save the image or close the image.
def picture():
    # Print instructions for the user
    print('\nInstructions:')
    print('Before starting, place the image that you would like to use in the same folder as this program.')
    print('While your image is displayed, press:')
    print('1) To save your image.')
    print('0) To close the image.\n')
    
    # Take in the filename from the user
    image_name = input('Enter the name + extension of your image (e.g. Test.jpg)')

    # Load the classifier and image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image_name)

    # Convert the image to a grayscale image to enable face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Place rectangles around the faces in the image 
    # TODO: Replace with emoji code
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Show the end result to the user
    cv2.imshow('img', img)

    # Detect valid key presses from the user and perform appropriate actions accordingly
    key = cv2.waitKey()
    if key == 0:
        cv2.destroyAllWindows()
    elif key == 1:
        cv2.imwrite('Output.jpg', img)
        
# This function prints the main menu.
def menu():
    print('\nWelcome to our face detection & replacement tool!')
    print('This program allows you to either select an existing image or stream your webcam.')
    print('Then, it will detect the faces in the image and replace them with a random emoji.')
    print('Finally, you will be able to save the end result.\n')
    print('What would you like to do:')
    print('1) Use an existing image.')
    print('2) Use your webcam.')
    print('0) Exit the program. \n')


############################
# Main code
############################
# Call the main menu.
menu()
# Take the users input, according to the options provided in the main menu.
# If the input is invalid, prompt the user to try again.
while True:
    try:
        choice = input('Enter your choice:')
        if choice == '1':
            picture()
            menu()
        elif choice == '2':
            webcam()
            menu()
        elif choice == '0':
            sys.exit()
        else:
            print('That was not a valid number. Try again.')
            continue
    except ValueError:
        print("That was not a number. Try again.")

