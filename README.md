## ASL Active Learning GUI

This project was to make an interactive way to practice your alphabet in sign language.  Some images of the GUI and the results are shown below.  

![ASL Detection GUI](Images/GUI_Interface.png)

![ASL Example 1](Images/Pic1.png)

![ASL Example 2](Images/Pic2.png)

![ASL Example 3](Images/Pic3.png)

![ASL Example 4](Images/Pic4.png)

### Implementation details

The YOLO object detection was pre-trained and can be found at "https://github.com/cansik/yolo-hand-detection"

A CNN was trained on a dataset obtained from Kaggle to predict what letter a hand was signing.  The YOLO hand detection was used to draw a box around the users hand so it could be cropped for the image analysis using the Tensorflow CNN model trained on the large Kaggle dataset.  

