import cv2
import numpy as np
import time
import autopy
import mediapipe as mp
import math
import pyautogui
import threading
import speech_recognition as sr

# ----------------- Hand Detector Class --------------------
class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if not self.lmList or len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            fingers.append(1 if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2] else 0)
        return fingers

    def findDistance(self, p1, p2, img=None, draw=True):
        try:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if draw and img is not None:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            return length, [x1, y1, x2, y2, cx, cy]
        except:
            return 0, [0, 0, 0, 0, 0, 0]

# ----------------- Voice Typing Function --------------------
def voice_typing_once():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening for voice typing...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"Original Voice Input: {text}")

            text = text.lower()

            # Define replacements
            replacements = {
                # Symbols
                "comma symbol": ",",
                "fullstop symbol": ".",
                "space symbol": " ",
                "question mark symbol": "?",
                "exclamation mark symbol": "!",
                "colon symbol": ":",
                "semicolon symbol": ";",
                "hashtag symbol": "#",
                "at the rate symbol": "@",
                "dollar symbol": "$",
                "percent symbol": "%",
                "ampersand symbol": "&",
                "star symbol": "*",
                "plus symbol": "+",
                "minus symbol": "-",
                "equal symbol": "=",
                "left bracket symbol": "(",
                "right bracket symbol": ")",
                "left square bracket symbol": "[",
                "right square bracket symbol": "]",
                "left curly bracket symbol": "{",
                "right curly bracket symbol": "}",
                "slash symbol": "/",
                "backslash symbol": "\\",
                "double quote symbol": "\"",
                "single quote symbol": "'",
                "greater than symbol": ">",
                "less than symbol": "<",
                # Emojis
                "smiley face emoji": "😊",
                "sad face emoji": "😢",
                "heart emoji": "❤️",
                "thumbs up emoji": "👍",
                "thumbs down emoji": "👎",
                "fire emoji": "🔥",
                "clap emoji": "👏",
                "thinking face emoji": "🤔",
                "crying face emoji": "😭",
                "angry face emoji": "😡",
                "star emoji": "⭐",
            }

            for word, symbol in replacements.items():
                text = text.replace(word, symbol)

            pyautogui.typewrite(text)

    except sr.WaitTimeoutError:
        print("Timeout: No speech detected.")
    except sr.UnknownValueError:
        print("Error: Could not understand audio.")
    except sr.RequestError:
        print("Error: Speech Recognition service unavailable.")
    finally:
        global voice_keyboard_running
        voice_keyboard_running = False   # after typing once, turn OFF

# ----------------- Main Program --------------------
wCam, hCam = 640, 480
frameR = 100
smoothening = 1.67
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
voice_keyboard_running = False

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        x1, y1 = lmList[8][1:]
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)

        # Gesture for mouse movement
        if fingers[1] == 1 and all(f == 0 for i, f in enumerate(fingers) if i != 1):
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

        # Left Click (Index + Middle)
        if fingers[1] == 1 and fingers[2] == 1:
            length, _ = detector.findDistance(8, 12, img)
            if length < 40:
                autopy.mouse.click()

        # Right Click (Index + Thumb)
        if fingers[1] == 1 and fingers[0] == 1:
            length, _ = detector.findDistance(8, 4, img)
            if length < 40:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)

        # Voice Keyboard Activation (Index + Middle + Ring fingers up)
        if fingers[4] == 1 and fingers[2] == 1 and fingers[3] == 1:
            if not voice_keyboard_running:
                voice_keyboard_running = True
                threading.Thread(target=voice_typing_once).start()

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
