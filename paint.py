# author : Trung Thanh Nguyen(Jimmy) | 09/12/2004  | ng.trungthanh04@gmail.com
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import random
import time
import pyttsx3
import threading
import torch.nn as nn
softmax = nn.Softmax()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine = pyttsx3.init()
engine.setProperty('rate',150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def thread_speak(text):
        threading.Thread(target = speak_text,args=(text,),daemon=True).start()
def speak_text(text):
    engine.say(text)
    engine.runAndWait()
def load_model(model_path):
    global device
    model = torch.load(model_path).to(device)
    model.eval()
    return model
def load_icon(image_path):
    icons = {}
    for path_icon in os.listdir(image_path):
        if path_icon.endswith("png"):
            class_name = path_icon.split(".")[0]
            icon = cv2.imread(os.path.join(image_path, path_icon), cv2.IMREAD_UNCHANGED)
            if icon is None:
                print(f"Failed to load icon: {path_icon}")
                continue
            if icon.shape[2] == 3:
                b, g, r = cv2.split(icon)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                icon = cv2.merge((b, g, r, alpha))
            icon = cv2.resize(icon, (50, 50))
            icons[class_name] = icon
    return icons
def predict(model,canvas):
    global device
    mask_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask_canvas = cv2.medianBlur(mask_canvas, 9)
    mask_canvas = cv2.GaussianBlur(mask_canvas, (5, 5), 0)
    x, y = np.nonzero(mask_canvas)
    if len(x) == 0 or len(y) == 0:
        return None
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    cropped_image = mask_canvas[min_x:max_x, min_y:max_y]
    cropped_image = cv2.resize(cropped_image, (28, 28))
    cropped_image = np.array(cropped_image, dtype=np.float32)[None, None, :, :] / 255.0
    cropped_image = torch.from_numpy(cropped_image).to(device)
    with torch.no_grad():
        predictions = model(cropped_image)
        prob = softmax(predictions)
    max_value, max_index = torch.max(prob, dim=1)
    return max_value.item(), max_index.item()
def overlay_icon(combined, icon, x, y):
    h, w, _ = icon.shape
    b, g, r, a = cv2.split(icon)
    overlay_color = cv2.merge((b, g, r))
    if y + h > combined.shape[0] or x + w > combined.shape[1]:
        return
    roi = combined[y:y + h, x:x + w]
    alpha = a / 255.0
    for c in range(3):
        roi[:, :, c] = (1.0 - alpha) * roi[:, :, c] + alpha * overlay_color[:, :, c]
    combined[y:y + h, x:x + w] = roi
def paint(model,classes):
    global device
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    icon_path = "./Icon_image"
    icons = load_icon(icon_path)
    canvas = None
    is_drawing = False
    start_time = None
    text = ""
    spoken_time = time.time()
    cap = cv2.VideoCapture(0)
    aim = classes[random.randint(0,len(classes)-1)]
    pre_result = None
    with mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if canvas is None:
                canvas = np.zeros_like(frame)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    x_index = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    y_index = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                    if is_drawing:
                        cv2.circle(canvas,(x_index,y_index),5,(255,255,255),-1)
                    mp_drawing.draw_landmarks(frame,hand_landmark,mp_hands.HAND_CONNECTIONS)
            combined = cv2.add(frame,canvas)
            cv2.putText(combined,"Your challenge is:  {}".format(aim), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 1,cv2.LINE_AA)
            cv2.putText(combined, "Guide:", (0,410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),1, cv2.LINE_AA)
            cv2.putText(combined, "Press [C] to delete your paint", (0, 426), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(207, 207, 207), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [P] to predict your paint", (0, 443), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (207, 207, 207), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [D] to start/stop painting", (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (207, 207, 207), 1, cv2.LINE_AA)
            cv2.putText(combined, "Press [A] to change your challenge", (0, 477), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (207, 207, 207), 1, cv2.LINE_AA)
            if canvas is not None and (time.time() - spoken_time >= 5):
                    result = predict(model,canvas)
                    if result:
                        max_value, max_index = result
                        class_name = classes[max_index]
                        spoken_time = time.time()
                        if class_name != aim and max_value *100 >20:
                            if class_name != pre_result:
                                pre_result = class_name
                                thread_speak("It's seemed like you are drawing: {}".format(class_name))
                                print("It's seemed like you are drawing :{}".format(class_name))
            if text and start_time:
                if time.time() - start_time <= 3:
                    cv2.putText(combined, text, (50 , 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    icon_text = text.split(":")[-1].strip()
                    if icon_text in icons:
                        icon = icons[class_name]
                        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0][0]
                        icon_x = 50 + text_width + 10
                        icon_y = 40
                        overlay_icon(combined, icon, icon_x, icon_y)
                else:
                    text = ""
                    start_time = None
                    canvas = np.zeros_like(frame)
            cv2.imshow("Draw",combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                is_drawing = not is_drawing
            elif key == ord("p"):
                is_drawing = False
                result_p = predict(model,canvas)
                if result_p:
                    max_value_p,max_index_p = result_p
                    class_name = classes[max_index_p]
                    text  = f"You are drawing:  {class_name}"
                    start_time = time.time()
                    print("You are drawing: {} with {} %".format(class_name,max_value_p*100))
                    if class_name == aim:
                        thread_speak("Oh no, It's {} , Correct!, You Win".format(class_name))
                        is_drawing = False
                        aim = classes[random.randint(0, len(classes) - 1)]
                        thread_speak("Your new challenge is: {}".format(aim))
            elif key == ord("c"):
                canvas = np.zeros_like(frame)
            elif key == 27:
                break
            elif key == ord("a"):
                aim = classes[random.randint(0, len(classes)-1)]
                thread_speak("Your new challenge is {}".format(aim))
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    model_path = "./train_model_QuickDraw/quickdraw"
    model = load_model(model_path)
    classes = ["Airplane","Angel","Apple","Axe","Bat","Book","Boomerang","Camera","Cup","Fish","Flower","Mushroom","Radio","Sun","Sword"]
    paint(model,classes)