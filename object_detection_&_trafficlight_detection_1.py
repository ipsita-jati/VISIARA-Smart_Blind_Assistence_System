from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
import numpy as np

# =========================
# SETTINGS
# =========================
VIDEO_PATH = "E:/ipsita_project/AI_smart_glass/video6.mp4"

last_detected_frame = None

# =========================
# LOAD MODELS
# =========================
model_coco = YOLO("yolov8n.pt")

model_custom = YOLO(
    "E:/ipsita_project/AI_smart_glass/best_fine2.pt",
    task="detect"
)

print("Models Loaded")

# =========================
# CLASS-WISE CONFIDENCE
# =========================
custom_conf = {
    "close door": 0.55,
    "open manhole": 0.45,
    "pothole": 0.10,
    "stair": 0.50,
    "steps": 0.30,
    "tree": 0.60,
    "wall": 0.35
}

# =========================
# SPEAK FUNCTION
# =========================
def speak(text):

    engine = pyttsx3.init()

    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)

    engine.say(text)

    engine.runAndWait()

    engine.stop()

# =========================
# VIDEO LOAD
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# SPEAK TIMER
# =========================
last_speak = 0
delay = 2

# =========================
# FRAME SKIP
# =========================
frame_count = 0
skip_frames = 2

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (512, 512))

    # =========================
    # FRAME SKIPPING
    # =========================
    frame_count += 1

    if frame_count % skip_frames != 0:

        if last_detected_frame is not None:
            cv2.imshow("Smart Glass System", last_detected_frame)
        else:
            cv2.imshow("Smart Glass System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # =========================
    # COCO DETECTION
    # =========================
    results1 = model_coco(
        frame,
        classes=[
            0,1,2,3,5,7,9,11,
            13,14,15,16,19,39,
            41,42,43,44,45,56,
            58,59,60,61,62,63,
            67,72,73,75
        ],
        conf=0.40
    )

    # =========================
    # CUSTOM DETECTION
    # =========================
    results2 = model_custom(
        frame,
        conf=0.10,
        iou=0.45,
        agnostic_nms=True
    )

    # =========================
    # DRAW DETECTIONS
    # =========================
    frame = results1[0].plot()
    frame = results2[0].plot(img=frame)

    last_detected_frame = frame.copy()

    h, w, _ = frame.shape

    center_objects = []

    # =========================
    # PROCESS OBJECTS
    # =========================
    for results, model in [
        (results1, model_coco),
        (results2, model_custom)
    ]:

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Object center
                cx = int((x1 + x2) / 2)

                # Object area
                area = (x2 - x1) * (y2 - y1)

                cls = int(box.cls[0])

                name = model.names[cls]

                conf = float(box.conf[0])

                # =========================
                # CLASS-WISE CONFIDENCE
                # =========================
                required_conf = custom_conf.get(name, 0.40)

                if conf < required_conf:
                    continue

                # =========================
                # REMOVE FALSE POTHOLE
                # =========================
                if name == "pothole":

                    w_box = x2 - x1
                    h_box = y2 - y1

                    pothole_area = w_box * h_box

                    # Ignore tiny potholes
                    if pothole_area < 1500:
                        continue

                # =========================
                # REMOVE FALSE MANHOLE
                # =========================
                if name == "open manhole" and conf < 0.45:
                    continue

                # =========================
                # TRAFFIC LIGHT DETECTION
                # =========================
                if name == "traffic light":

                    light_crop = frame[y1:y2, x1:x2]

                    if light_crop.size > 0:

                        light_crop = cv2.resize(light_crop, (60, 120))

                        h_crop = light_crop.shape[0]

                        top = light_crop[0:h_crop//3, :]
                        middle = light_crop[h_crop//3:2*h_crop//3, :]
                        bottom = light_crop[2*h_crop//3:h_crop, :]

                        hsv_top = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
                        hsv_middle = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
                        hsv_bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)

                        # RED
                        lower_red1 = np.array([0, 120, 70])
                        upper_red1 = np.array([10, 255, 255])

                        lower_red2 = np.array([170, 120, 70])
                        upper_red2 = np.array([180, 255, 255])

                        red_mask1 = cv2.inRange(
                            hsv_top,
                            lower_red1,
                            upper_red1
                        )

                        red_mask2 = cv2.inRange(
                            hsv_top,
                            lower_red2,
                            upper_red2
                        )

                        red_mask = red_mask1 + red_mask2

                        # GREEN
                        lower_green = np.array([40, 40, 40])
                        upper_green = np.array([90, 255, 255])

                        green_mask = cv2.inRange(
                            hsv_bottom,
                            lower_green,
                            upper_green
                        )

                        # YELLOW
                        lower_yellow = np.array([15, 150, 150])
                        upper_yellow = np.array([30, 255, 255])

                        yellow_mask = cv2.inRange(
                            hsv_middle,
                            lower_yellow,
                            upper_yellow
                        )

                        red_pixels = cv2.countNonZero(red_mask)
                        green_pixels = cv2.countNonZero(green_mask)
                        yellow_pixels = cv2.countNonZero(yellow_mask)

                        traffic_signal = None

                        threshold = 40

                        if (
                            red_pixels > threshold and
                            red_pixels > green_pixels and
                            red_pixels > yellow_pixels
                        ):

                            traffic_signal = "Traffic light detected. Red light"

                        elif (
                            green_pixels > threshold and
                            green_pixels > red_pixels and
                            green_pixels > yellow_pixels
                        ):

                            traffic_signal = "Traffic light detected. Green light"

                        elif (
                            yellow_pixels > threshold and
                            yellow_pixels > red_pixels and
                            yellow_pixels > green_pixels
                        ):

                            traffic_signal = "Traffic light detected. Yellow light"

                        if traffic_signal:

                            cv2.putText(
                                frame,
                                traffic_signal,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                            current_time = time.time()

                            if current_time - last_speak >= 3:

                                print(traffic_signal)

                                threading.Thread(
                                    target=speak,
                                    args=(traffic_signal,),
                                    daemon=True
                                ).start()

                                last_speak = current_time

                # =========================
                # CENTER AREA ONLY
                # =========================
                if (
                    w * 0.28 < cx < w * 0.72
                    and
                    name != "traffic light"
                ):

                    center_objects.append({
                        "name": name,
                        "area": area
                    })

    # =========================
    # NEAREST CENTER OBJECT
    # =========================
    if center_objects:

        nearest = max(
            center_objects,
            key=lambda x: x["area"]
        )

        current_object = nearest["name"]

        nearest_area = nearest["area"]

        current_time = time.time()

        # =========================
        # WARNING ONLY WHEN CLOSE
        # =========================
        if nearest_area > 12000:

            if current_time - last_speak >= delay:

                print(f"Warning {current_object} ahead")

                threading.Thread(
                    target=speak,
                    args=(f"{current_object} ahead",),
                    daemon=True
                ).start()

                last_speak = current_time

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("Smart Glass System", frame)

    # ESC TO EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()

cv2.destroyAllWindows()