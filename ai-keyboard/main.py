""" an ai powered virtual keyboard controlled by our hands """

# step 1 :- capture the made by parthiv frames by using opencv

import cv2             # opencv is a module which is used to interact and perform task using camera
import numpy as np     # numpy is a scientific module wwhich is used to do operation based on array 
import mediapipe as mp # mediapipe is an ai module developed by google for detection systems 
import time            # time module to get the time
from pynput.keyboard import Controller # pynput is the module used for the automation of the keyboard

# defining a class for the key's representation in the keyboard

class Key():           
    def __init__(self, x, y, w, h, text): # defining a init function to make a looping function to pass out the variables
        self.x = x # giving the postion value of the x axis of the key
        self.y = y # giving the postion value of the y axis of the key
        self.w = w # giving the postion value of the width of the key
        self.h = h # giving the postion value of the height of the key
        self.text = text # text is the name of the key to pe presented
    
    # drefining a function to draw the key's of the keyboard on the opencv frame

    def drawKey(self, img, text_color=(255,255,255), bg_color=(0,0,0), alpha=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]  # this extract the region of the image where the key is drawn
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8) * bg_color     # creates a new image of the same size as the key filled with  the bg 
        white_rect = white_rect.astype(np.float32)  # Convert white_rect to float32 type array
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 0.0, dtype=cv2.CV_8U)  # binds the new image of key with the previous one and control the transparency (alpha)
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res     # the region of the new binded key depicted in the frame

        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness) # calculates the size of the key text
        text_pos = (int(self.x + self.w/2 - text_size[0][0]/2), int(self.y + self.h/2 + text_size[0][1]/2)) # manages the padding inside the keys to center out the text placed in the key
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)   # drwad the text in the key 

    # defining a function to check the key and postion of the keys

    def isOver(self, x, y):
        if (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y):
            return True # checks the x coordinates and y coordinates is within the vertical bounds of the key if yes condition set to true
        return False # otherwise condition set to false


# defining a new clas for the hand tracking

class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): # setting up the constraints for the tracking 
        self.mode = mode # setting up the initial value
        self.maxHands = maxHands # setting up the initial value
        self.detectionCon = detectionCon # setting up the initial value
        self.trackCon = trackCon # setting up the initial value

        self.mpHands = mp.solutions.hands # imports the hand detection algorithm from mediapipe
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon) # pasess on the constraint value for detection
        self.mpDraw = mp.solutions.drawing_utils # draws the points/landmarks upon the hand in the frames collected using mediapipe

    def findHands(self, img, draw=True): # defining a function to detetect the hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converts the frame to rgb format
        self.results = self.hands.process(imgRGB)     # process the frames by passing the frames to the hand detecting model 

        if self.results.multi_hand_landmarks: # a condition statement to check whether a hand is detected 
            for handLm in self.results.multi_hand_landmarks: # for loop to iterate through all the hands detected in the frame 
                if draw:  # if the draw statement is true that is draw will be true if a hand is detected in the frame
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)  # draws all the ladmarks on the frame 
        return img  # returns the image of the frame with the drawn landmarks 
 
    def getPostion(self, img, handNo=0, draw=True): # this method gets the position of the landmarks for the specific hand in the image 
        lmList =[] # an empty list to store the position data of the landmarks 
        if self.results.multi_hand_landmarks: # condition statement to check for the landmarks
            myHand = self.results.multi_hand_landmarks[handNo] # storing the landmarks in the myhand variable
            for id, lm in enumerate(myHand.landmark): # iterating through all the landmarks in the frame and updating its id and position
                h, w, c = img.shape # retreving the data from the frames like height width and colour channels 
                cx, cy = int(lm.x*w), int(lm.y*h) # a simple expression to match the position of the landmarls into an integer for the simplicity of taking values 
                lmList.append([id, cx, cy]) # this appends/add the id value and the x and y values of the landmarks to the empty list created before 

                if draw: # if the draw mode is true
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED) # cv2 will draw the circle points upon each landmark
        return lmList # returns the list of the position values of the landmarks in the frame 

def getMousPos(event, x, y, flags, param): # defining a function to handle the mouse events 
    global clickedX, clickedY # setting the mouse clickes x axis and mouse clickes y axis variables as global variables 
    global mouseX, mouseY # setting the mousex axis and mouse y axis variables as global variables 
    if event == cv2.EVENT_LBUTTONUP: # an event handling condiiton statement to check whether the mouse is clicked
        clickedX, clickedY = x, y # if clickes the global variables are updated accordingly 
    if event == cv2.EVENT_MOUSEMOVE: # an event handling condition statement to check whether the mouse is moved 
        mouseX, mouseY = x, y # if the mouse if moved then the gloabal variables are updated accordingly 

def calculateIntDidtance(pt1, pt2): # defining a function to calculate the distance between the points of the two reference bodies 
    return int(((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5) # euclidean distance formula is used to find the distance between the reference points 
 
# Define keyboard keys and layout
keys = [] # an empty list to store the values of the keys 
letters = list("QWERTYUIOPASDFGHJKLZXCVBNM") # creating a list calles letters to store the keys to be displayed 
w, h = 80, 60 # setting the width and height of the keys 
startX, startY = 40, 200 # setting the initial coordinates of the key to be placed

for i, l in enumerate(letters):  # an iteration loop that iterated through all the letters in the letters list and l is the number of letters stored in the letterlist
    if i < 10:                   # i is the number of the iteration being performed
        keys.append(Key(startX + i*w + i*5, startY, w, h, l)) # according to the iterations the value of the key is processed and appended to the key list 
    elif i < 19:                 # if the iterative value or the value of i is greater than 10 and less than 19
        keys.append(Key(startX + (i-10)*w + i*5, startY + h + 5, w, h, l))   # accordingly to the iteration the value of the key is processed and appended to the key list 
    else:                        # a else statement to process the iterative value above 19 
        keys.append(Key(startX + (i-19)*w + i*5, startY + 2*h + 10, w, h, l)) # accordingly to the iteration the value of the key is processed and appended to the key list 

keys.append(Key(startX + 25, startY + 3*h + 15, 5*w, h, "Space")) # adding the space bar key to the visual keyboard
keys.append(Key(startX + 8*w + 50, startY + 2*h + 10, w, h, "clr")) # adding the clear key to the visual keyboard
keys.append(Key(startX + 5*w + 30, startY + 3*h + 15, 5*w, h, "<--")) # adding the backspace key to the visual keyboard

showKey = Key(300, 5, 80, 50, 'Show') # creating the show button in the frame
exitKey = Key(300, 65, 80, 50, 'Exit') # creating the exit button in the frame 
textBox = Key(startX, startY-h-5, 10*w+9*5, h, '') # creating a text box to show the key pressed in real time in the frame

cap = cv2.VideoCapture(0) # using opencv2 module to initialize the process of made by parthiv recording 
tracker = HandTracker(detectionCon=1) # hand tracker object to track hands in the made by parthiv 


frameHeight, frameWidth, _ = cap.read()[1].shape # getting the frames height and width from the opencv
showKey.x = int(frameWidth*1.5) - 85 # setting the position value of the show button according to the height and width of the screen 
exitKey.x = int(frameWidth*1.5) - 85 # setting the position value of the exit button according to the height and width of the screen 

cv2.namedWindow('made by parthiv')  # setting the title of the window

clickedX, clickedY = 0, 0 # initial values for the cliked x axis and clicked y axis
mousX, mousY = 0, 0 # initial values for the mouse x axis and miuse y axis 

counter = 0 # initial value of the counter is set to 0 
show = False # initial value of show is also set to true
ptime = time.time() # getting the current time and assighining it to the ptime variable 
previousClick = 0 # setting the value of the previosclick to 0 
keyboard = Controller() # controller is the object which ios used to send keystrokes to the keyboard

while True: # opening an infinite loop for the capturing of the frames from the camera 
    if counter > 0: # counter is used to control the event of analysing the frames 
        counter -= 1

    signTipX = 0 # setting the x position vakue of the index fingertip to 0
    signTipY = 0 # setting the y position value of the index finger to 0 

    thumbTipX = 0 # setting the x position vakue of the thump fingertip to 0
    thumbTipY = 0 # setting the y position vakue of the thump  fingertip to 0

    # checking whether there is a frame being recorded by the opencv

    ret, frame = cap.read()
    if not ret: # if there are no frames detected then the program stops 
        break
    frame = cv2.resize(frame, (int(frameWidth*1.5), int(frameHeight*1.5))) # resizing the frames recorded by the webcam by 1.5 times 
    frame = cv2.flip(frame, 1) # flipping the frames captured by the opencv
    
    frame = tracker.findHands(frame) # detects the hands from the frame by using the hand objects
    lmList = tracker.getPostion(frame, draw=False) # get the position values of the lanmdmarks
    if lmList: # if the landmarks are detetcted 
        signTipX, signTipY = lmList[8][1], lmList[8][2] # captures the x axis and y axis of the pointy finger 
        thumbTipX, thumbTipY = lmList[4][1], lmList[4][2] # cpatures the x axis and y axis of the thumb finger 
        if calculateIntDidtance((signTipX, signTipY), (thumbTipX, thumbTipY)) < 50: # checks whether the distance bertween the two reference bodies are less than 50 
            centerX = int((signTipX+thumbTipX)/2) # if the distance is less than 50 then the x axis of the center point of distance between them is calculated 
            centerY = int((signTipY + thumbTipY)/2) # if the distance is less than 50 then the y aixs of the center point of distance between them is calculated 
            cv2.line(frame, (signTipX, signTipY), (thumbTipX, thumbTipY), (255, 0, 0), 2) # then a straight line is drawn between them 
            cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), cv2.FILLED) # a circle is also drawn at the middle part of the staright line drawn


    cv2.setMouseCallback('made by parthiv', getMousPos) # setting a event handling to call the mouse event handler whenevr there is a detection of mouse movements 


    ctime = time.time() # getting the time 
    fps = int(1/(ctime - ptime)) # calculating fps the fps is calculated on the basis of the time difference between the previous frame and the current frame 
    
    cv2.putText(frame, str(fps) + " FPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2) # draws the fps in the frame 
    showKey.drawKey(frame, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.5) # draws the show key
    exitKey.drawKey(frame, (255, 255, 255), (0, 0, 0), 0.1, fontScale=0.5) # draws the exit key 
    cv2.namedWindow('made by parthiv') # creates a named window made by parthiv 
    cv2.setMouseCallback('made by parthiv', getMousPos) # setting a mousecall back function 

    if showKey.isOver(clickedX, clickedY): # checking if the show button is clicked 
        show = not show # setting the variables of the show its an inverse event handle such as if the show was true upon clicking it is turnes to false and if false at start it changes to true
        showKey.text = "Hide" if show else "Show" # if the value of show false button changes to hide and in true then chaneges to show 
        clickedX, clickedY = 0, 0 # resets the clicked x value and clicked y value to 0,0

    if exitKey.isOver(clickedX, clickedY): # if the exit is clicked or not 
        break # if clicked breaks and enda the code 

    alpha = 0.5 # transparency variable 
    if show: # checks whether the show is TRUE or not 
        textBox.drawKey(frame, (255, 255, 255), (0, 0, 0), 0.3) # draws the textbox on the frame 
        for k in keys: # iterates throught the keys stored in key list 
            if k.isOver(mouseX, mouseY) or k.isOver(signTipX, signTipY): #checks whether the position of the index finger is upon the position of the key 
                alpha = 0.1 # if the position of the mouse is upon the key then the transparency is set to 0.1
                if k.isOver(clickedX, clickedY): # checks whether any keys were pressed                          
                    if k.text == '<--':          # if the key pressed is backspace
                        textBox.text = textBox.text[:-1] # the last text in the textbox is deleted 
                    elif k.text == 'clr': # if the key clicked is the clear button 
                        textBox.text = '' # it clears all the text stored in the textbox
                    elif len(textBox.text) < 30: # if the length of the characters in the textbox is less than 30 characters then the key appended to the textbox
                        if k.text == 'Space': # if the space key is clicked
                            textBox.text += " " # it adds a space between the text
                        else:                      # else no special keys are clicked then 
                            textBox.text += k.text # the clicked key is added to the textbox
                            
                if (k.isOver(thumbTipX, thumbTipY)): # checks whether the thumb finger is abive any of the keys 
                    clickTime = time.time() # gets the current time when the thumb if above the keys 
                    if clickTime - previousClick > 0.4:  # checks when was the last time a click was registers is it less than o.4 seconds                              
                        if k.text == '<--': # if the key pressed is backspace
                            textBox.text = textBox.text[:-1] # the last text in the textbox is deleted 
                        elif k.text == 'clr': # if the key clicked is the clear button 
                            textBox.text = '' # it clears all the text stored in the textbox
                        elif len(textBox.text) < 30: # if the length of the characters in the textbox is less than 30 characters then the key appended to the textbox
                            if k.text == 'Space': # if the space key is clicked
                                textBox.text += " " # it adds a space between the text
                            else:                   # else no special keys are clicked then 
                                textBox.text += k.text # the clicked key is added to the textbox
                                keyboard.press(k.text) # stimulates a keystorke to the keyboard 
                        previousClick = clickTime # updates the clicktime to the current time as a new click is registered
            k.drawKey(frame, (255, 255, 255), (0, 0, 0), alpha=alpha) # draws the key on the frame 
            alpha = 0.5 # sets the transparency to 0.5
        clickedX, clickedY = 0, 0        # resets the clicked x value and clicked y value to 0,0
    ptime = ctime # sets the previous time to current time 
    cv2.imshow('made by parthiv', frame) # show all the frames captured by the opencv
  
    pressedKey = cv2.waitKey(1) # waits to see whether a key is pressed
    if pressedKey == ord('q'): # if the q key is pressed the process stops and 
        break                  # breaks

cap.release() # releases the camera and retrieves all the rights of the opencv
cv2.destroyAllWindows() # destory all the windows 
