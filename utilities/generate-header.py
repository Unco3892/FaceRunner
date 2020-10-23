# This file was used to generate the header which was not palced directly in the main body as this method leaves some newlines which we desired to remove.

from pyfiglet import Figlet

header = Figlet(font='big')
subheader = Figlet(font='digital')
print(header.renderText("FaceRunner"))
print(subheader.renderText("By Ilia Azizi & Emile Evers"))