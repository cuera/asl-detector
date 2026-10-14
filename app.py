import pickle
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

model = pickle.load(open('./model.p', 'rb'))['model']
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
cap = None

root = tk.Tk()
root.title('ASL Detector')
root.configure(bg='#111')
video = tk.Label(bg='#111')
video.pack(padx=10, pady=10)
letter = tk.Label(text='-', font=('Segoe UI', 64, 'bold'), fg='#fff', bg='#111')
letter.pack()
btn = tk.Button(font=('Segoe UI', 12), relief='flat', bg='#2d6cdf', fg='#fff', width=12)
btn.pack(pady=10)


def loop():
    if cap is None:
        return
    ok, frame = cap.read()
    if ok:
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = '-'
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            xs, ys = [p.x for p in lm], [p.y for p in lm]
            feats = [v for p in lm for v in (p.x - min(xs), p.y - min(ys))]
            text = chr(65 + int(model.predict([np.asarray(feats)])[0]))
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (int(min(xs) * w) - 10, int(min(ys) * h) - 10),
                          (int(max(xs) * w) + 10, int(max(ys) * h) + 10), (255, 255, 255), 2)
        letter.config(text=text)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video.imgtk = img
        video.config(image=img)
    root.after(15, loop)


def toggle():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        btn.config(text='Stop')
        loop()
    else:
        cap.release()
        cap = None
        video.config(image='')
        letter.config(text='-')
        btn.config(text='Start')


btn.config(text='Start', command=toggle)
root.protocol('WM_DELETE_WINDOW', lambda: (cap and cap.release(), root.destroy()))
root.mainloop()
