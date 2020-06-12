from tkinter import *
import string
from PIL import ImageTk, Image
from collections import OrderedDict
import webbrowser
from tkhtmlview import HTMLLabel
import cv2
import numpy as np
from yolo import YOLO 
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import random 


l=list(string.ascii_uppercase)

import argparse
import cv2

from yolo import YOLO

root = Tk()
root.title("ASL")
path="sign.png"
# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)

# Create a Tkinter variable
tkvar = StringVar(root)

# Dictionary with options
choices=OrderedDict([(cc,i) for i,cc in enumerate(l)])
c=OrderedDict([('Random',-1)])
c.update(choices)
choices=c
k=sorted(choices.keys())
tkvar.set('Random') # set the default option
popupMenu = OptionMenu(mainframe, tkvar, *k)
Label(mainframe, text="Choose the alphabet").grid(row = 2, column = 0)
popupMenu.grid(row = 3, column =0)

photo1=PhotoImage(file=path)
Label(root,image=photo1,bg="black").grid(row=2,column=0) #E=East,W=West

# on change dropdown value
def press():
    new_model = load_model('Letter_Model_V3_999_1')
    alph = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
    alph_dict = {}
    for i,n in enumerate(alph):
        alph_dict.update({i:n})
    
    camera=cv2.VideoCapture(0)
    img_counter = 0
    List_alph = [i for i in alph]
    if tkvar.get()=='Random':
        letter = random.choice(List_alph)
    else:
        letter=tkvar.get()
    while True:
        ret, frame = camera.read()
        if not ret:
            print("failed to grab frame")
            break
        color = (0, 255, 255)
        cv2.putText(frame,'Please try to sign the letter: ' + (letter),(20,20),cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        cv2.imshow("Gesture Detector", frame)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            print("{} written!".format(img_name))
            x=frame
            img_counter += 1
            
            yolo = YOLO("/Users/Denny/Desktop/Hack_The_Northeast/yolo-hand-detection/models/cross-hands.cfg", "/Users/Denny/Desktop/Hack_The_Northeast/yolo-hand-detection/models/cross-hands.weights", ["hand"])
            width, height, inference_time, results = yolo.inference(x)
            frame = x        
            for detection in results:
                id, name, confidence, x, y, w, h = detection
                cx = x + (w / 2)
                cy = y + (h / 2)
                crop_img = frame[y-50:y+h+50, x-50:x+w+50]
                im = Image.fromarray(crop_img)
                im.save("your_file.png")
                im = cv2.imread('your_file.png',0)
                new_img = cv2.resize(im,(28,28))
                the_class = new_model.predict_classes(new_img.reshape(1,28,28,1))
                Answer = alph_dict[the_class[0]]
                # draw a bounding box rectangle and label on the image
                color = (0, 255, 255)

                if Answer == letter:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                    text = 'Correct Answer for: ' + Answer
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
                    text = 'Try again!'
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 2)
                    
            cv2.imshow("preview", frame)

    camera.release()
    
    cv2.destroyAllWindows()

w = Label(root, text="Learn Sign Language",font=('bold')).grid(row = 8, column = 0)
t=HTMLLabel(root, html='<a href="https://www.youtube.com/watch?v=Raa0vBXA8OQ"> Tutorial </a>',font=("Courier", 1)).grid(row=9,column=0)

B = Button(root, text ="Practice", command = press,font=("Times New Roman", 30)).grid(row=5,column=0)

w = Label(root, text="ASL Practice Arena",font=("Times New Roman", 20,'bold')).grid(row = 1, column = 0)

Rules='Instructions:\nChoose the practice letter\n1. Click the button when you are ready to go\n2. Capture using spacebar\n3. Exit the window\n4. See the result\n5.Exit by escape'

w = Label(root, text=Rules,font=("Times New Roman", 10)).grid(row = 6, column = 0)

root.mainloop()
