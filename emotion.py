import sys,os,cv2,time
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QAction, qApp, QMainWindow
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5 import *


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        loadUi('graph.ui', self)
        
        self.pushButton.clicked.connect(self.start)
        self.pushButton_2.clicked.connect(self.closed)
    def closed(self):
        self.close()
    def start(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._frame)
        self.timer.start(4)

    def _frame(self):
        ret, self.image = self.capture.read()
        faceCascade = cv2.CascadeClassifier(
            'Cascade/haarcascade-frontalface-default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #Dictionary for emotion recognition model output and emotions
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                    3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

        #Upload images of emojis from emojis folder
        emoji = []
        for index in range(6):
            emotion = emotions[index]
            emoji.append(cv2.imread('emoji/' + emotion + '.png', -1))
        model = load_model('models/emotion.h5', compile=False)
        frame_count = 0
        

        if ret:
            
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            #Face detection takes approx 0.07 seconds
            start_time = time.time()
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(100, 100),)
            
            y0 = 15
            for index in range(6):
                emotions[index] + ': ', (5, y0)
                y0 += 15

            try:
                # Flag for showing probability graph of only one face
                FIRSTFACE = True
                if len(faces) > 0:
                    for x, y, width, height in faces:
                        cropped_face = gray[y:y + height, x:x + width]
                        test_image = cv2.resize(cropped_face, (48, 48))
                        test_image = test_image.reshape([-1, 48, 48, 1])

                        test_image = np.multiply(test_image, 1.0 / 255.0)

                        # Probablities of all classes
                        #Finding class probability takes approx 0.05 seconds
                        start_time = time.time()
                        if frame_count % 5 == 0:
                            probab = model.predict(test_image)[0] * 100
                            
                            label = np.argmax(probab)
                            probab_predicted = int(probab[label])
                            predicted_emotion = emotions[label]
                            
                            frame_count = 0
                            
                           

                        frame_count += 1
                       
                        # Drawing on frame
                        font_size = width / 300
                        filled_rect_ht = int(height / 5)

                        #Resizing emoji according to size of detected face
                        emoji_face = emoji[(label)]
                        emoji_face = cv2.resize(
                            emoji_face, (filled_rect_ht, filled_rect_ht))

                        #Positioning emojis on frame
                        emoji_x1 = x + width - filled_rect_ht
                        emoji_x2 = emoji_x1 + filled_rect_ht
                        emoji_y1 = y + height
                        emoji_y2 = emoji_y1 + filled_rect_ht

                        #Drawing rectangle and showing output values on frame
                        cv2.rectangle(self.image, (x, y), (x + width,
                                                      y + height), (155, 155, 0), 2)
                        cv2.rectangle(self.image, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                                      (155, 155, 0), cv2.FILLED)
                        cv2.putText(self.image, predicted_emotion+' ' + str(probab_predicted)+'%',
                                    (x, y + height + filled_rect_ht-10), font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        # Showing emoji on frame
                        for c in range(0, 3):
                            self.image[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + self.image[emoji_y1:emoji_y2, emoji_x1:emoji_x2, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                            
            except Exception as error:

                pass
        
        
        self.display(self.image, 1)

    def display(self, img, windows=1):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outimage = QtGui.QImage(
            img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outimage = outimage.rgbSwapped()
        if windows == 1:
            self.label.setPixmap(QtGui.QPixmap.fromImage(outimage))
            self.label.setScaledContents(True)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("LabVision")
    window.show()
    sys.exit(app.exec_())
