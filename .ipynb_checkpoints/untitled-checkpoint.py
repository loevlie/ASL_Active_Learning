import cv2
import numpy as np
from yolo import YOLO 
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import random 

new_model = load_model('Letter_Model_V3_965_1')
alph = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
alph_dict = {}
for i,n in enumerate(alph):
    alph_dict.update({i:n})

camera=cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
List_alph = [i for i in alph]
letter = random.choice(List_alph)
while True:
    ret, frame = camera.read()
    if not ret:
        print("failed to grab frame")
        break
    color = (0, 255, 255)
    cv2.putText(frame,letter,(20,20),cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    yolo = YOLO("/Users/Denny/Desktop/Hack_The_Northeast/yolo-hand-detection/models/cross-hands.cfg", "/Users/Denny/Desktop/Hack_The_Northeast/yolo-hand-detection/models/cross-hands.weights", ["hand"])
    width, height, inference_time, results = yolo.inference(frame)
        
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)
        crop_img = frame[y-50:y+h+50, x-50:x+w+50]
            
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(0)
        im = Image.fromarray(crop_img)
        im.save("your_file.png")
        im = image.load_img('your_file.png',target_size=(28,28),color_mode='grayscale')
        new_img = image.img_to_array(im)
        the_class = new_model.predict_classes(new_img.reshape(1,28,28,1))
        Answer = alph_dict[the_class[0]]
        # draw a bounding box rectangle and label on the image
        color = (255, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #text = "%s (%s)" % (name, round(confidence, 2))
        text = Answer
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)
    cv2.imshow("test", frame)

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
            
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)
            im = Image.fromarray(crop_img)
            im.save("your_file.png")
            im = image.load_img('your_file.png',target_size=(28,28),color_mode='grayscale')
            new_img = image.img_to_array(im)
            the_class = new_model.predict_classes(new_img.reshape(1,28,28,1))
            print(the_class)
            Answer = alph_dict[the_class[0]]
            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #text = "%s (%s)" % (name, round(confidence, 2))
            text = Answer
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
        cv2.imshow("preview", frame)
        

camera.release()

cv2.destroyAllWindows()
print(Answer)