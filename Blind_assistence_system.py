from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
import numpy as np
from collections import defaultdict, deque

# =========================
# SETTINGS
# =========================
VIDEO_PATH = "video_materials/bagnan_highroad_real_time.mp4"

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
# SPEAK FUNCTION
# =========================
def speak(text):

    def run():

        engine = pyttsx3.init()

        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)

        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(
        target=run,
        daemon=True
    ).start()

# =========================
# INDOOR OUTDOOR DETECTION
# =========================
def detect_environment(frame, results1):

    h, w, _ = frame.shape

    # -------------------------
    # BRIGHTNESS
    # -------------------------
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    # -------------------------
    # SKY DETECTION
    # -------------------------
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    lower_sky = np.array([90, 40, 40])
    upper_sky = np.array([130, 255, 255])

    sky_mask = cv2.inRange(
        hsv,
        lower_sky,
        upper_sky
    )

    sky_pixels = cv2.countNonZero(
        sky_mask
    )

    # -------------------------
    # EDGE DENSITY
    # -------------------------
    edges = cv2.Canny(
        gray,
        80,
        150
    )

    edge_pixels = cv2.countNonZero(
        edges
    )

    # -------------------------
    # OBJECT ANALYSIS
    # -------------------------
    outdoor_score = 0
    indoor_score = 0

    outdoor_objects = [
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "traffic light",
        "tree",
        "bench"
    ]

    indoor_objects = [
        "chair",
        "bed",
        "refrigerator",
        "toilet",
        "dining table",
        "potted plant"
    ]

    for r in results1:

        for box in r.boxes:

            cls = int(box.cls[0])

            name = model_coco.names[cls]

            conf = float(box.conf[0])

            if conf < 0.50:
                continue

            if name in outdoor_objects:
                outdoor_score += 2

            if name in indoor_objects:
                indoor_score += 2

    # -------------------------
    # SKY BOOST
    # -------------------------
    if sky_pixels > 7000:
        outdoor_score += 4

    # -------------------------
    # BRIGHTNESS
    # -------------------------
    if brightness > 150:
        outdoor_score += 1

    # -------------------------
    # EDGE DENSITY
    # -------------------------
    if edge_pixels > 25000:
        indoor_score += 2

    # -------------------------
    # FINAL DECISION
    # -------------------------
    if outdoor_score > indoor_score:
        return "Outdoor"

    else:
        return "Indoor"
    
# =========================
# STAIR ANALYSIS
# =========================
def analyze_stairs(stair_crop):

    try:

        gray = cv2.cvtColor(
            stair_crop,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=10
        )

        step_count = 1
        stair_type = "stairs"

        if lines is not None:

            horizontal_lines = []

            for line in lines:

                x1, y1, x2, y2 = line[0]

                angle = abs(
                    np.degrees(
                        np.arctan2(
                            y2 - y1,
                            x2 - x1
                        )
                    )
                )

                # Horizontal stair edges
                if angle < 10:

                    horizontal_lines.append(
                        (x1, y1, x2, y2)
                    )

            # Remove duplicate lines
            filtered_y = []

            for l in horizontal_lines:

                y_avg = (l[1] + l[3]) // 2

                if all(
                    abs(y_avg - yy) > 12
                    for yy in filtered_y
                ):

                    filtered_y.append(y_avg)

            # Step estimation
            step_count = max(
                1,
                round(len(filtered_y) / 2)
            )

            # =========================
            # Upward / Downward estimation
            # =========================
            if len(filtered_y) >= 3:

                filtered_y.sort()

                gaps = []

                for i in range(len(filtered_y) - 1):

                    gap = filtered_y[i+1] - filtered_y[i]

                    gaps.append(gap)

                mid = len(gaps) // 2

                top_gap = np.mean(gaps[:mid])

                bottom_gap = np.mean(gaps[mid:])

                # Perspective logic
                # Bigger gaps at bottom = upward stairs
                if bottom_gap > top_gap:

                    stair_type = "upward stairs"

                else:

                    stair_type = "downward stairs"

            else:

                stair_type = "stairs"

        return step_count, stair_type

    except Exception as e:

        print("Stair analysis error:", e)

        return 1, "stairs"

# =========================
# SAFE DIRECTION
# =========================
def get_safe_direction(frame, boxes):

    h, w, _ = frame.shape

    left_danger = 0
    right_danger = 0

    for box in boxes:

        x1, y1, x2, y2 = box

        area = (x2 - x1) * (y2 - y1)

        if area < 2500:
            continue

        cx = (x1 + x2) // 2

        if cx < w // 2:
            left_danger += 1
        else:
            right_danger += 1

    if left_danger < right_danger:
        return "Move left"

    elif right_danger < left_danger:
        return "Move right"

    else:
        return "Walk carefully"

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# STARTUP DELAY FOR SCREEN RECORDING
# =========================
ret, first_frame = cap.read()

if ret:

    first_frame = cv2.resize(
        first_frame,
        (416, 416)
    )

    cv2.imshow(
        "Smart Glass System",
        first_frame
    )

    cv2.waitKey(1)

    print("Starting in 5 seconds...")

    time.sleep(5)

    # Video restart from beginning
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

# =========================
# VARIABLES
# =========================
last_detected_frame = None

prev_time = time.time()

last_object_name = ""
last_object_time = 0

last_speak = 0
delay = 3

frame_count = 0
skip_frames = 3
# =========================
# ENVIRONMENT TRACKING
# =========================
last_environment = ""

# =========================
# VEHICLE MOTION TRACKING
# =========================
vehicle_history = defaultdict(
    lambda: deque(maxlen=3)
)

VEHICLE_CLASSES = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "person"
]

AREA_GROWTH_THRESHOLD = 1.05
SIDE_MOVEMENT_THRESHOLD = 25

BOTTOM_REGION = 0.50
# TTC SETTINGS
FAST_APPROACH_THRESHOLD = 1.20
MEDIUM_APPROACH_THRESHOLD = 1.08

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (416, 416))

    original_frame = frame.copy()

    frame_count += 1

    if frame_count % skip_frames != 0:

        if last_detected_frame is not None:
            cv2.imshow(
                "Smart Glass System",
                last_detected_frame
            )
        else:
            cv2.imshow(
                "Smart Glass System",
                frame
            )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # =========================
    # COCO DETECTION
    # =========================
    results1 = model_coco(
        frame,
        classes=[
            0, 1, 2, 3, 5, 7, 9,
            13, 14, 15, 16,
            39, 45,
            56, 58, 59,
            60, 61, 67, 72,
            73, 75
        ],
        conf=0.55
    )

    # =========================
    # CUSTOM DETECTION
    # =========================
    results2 = model_custom(
        frame,
        conf=0.30,
        iou=0.45,
        agnostic_nms=False
    )

    # =========================
    # DETECT ENVIRONMENT
    # =========================

    # Detect current frame environment
    current_environment = detect_environment(
        frame,
        results1
    )

    # Create history only once
    if 'environment_history' not in globals():

        environment_history = deque(maxlen=15)

    # Save current result
    environment_history.append(
        current_environment
    )

    # Stable counting
    outdoor_count = environment_history.count(
        "Outdoor"
    )

    indoor_count = environment_history.count(
        "Indoor"
    )

    # Final stable environment
    if outdoor_count >= indoor_count:

        environment = "Outdoor"

    else:

        environment = "Indoor"

    # Speak only when changed
    if environment != last_environment:

        print(f"{environment} detected")

        threading.Thread(
            target=speak,
            args=(f"{environment} detected",),
            daemon=True
        ).start()

        last_environment = environment

    # Draw detections
    frame = results1[0].plot()

    frame = results2[0].plot(
        img=frame
    )

    last_detected_frame = frame.copy()

    h, w, _ = frame.shape

    center_objects = []

    all_boxes = []

    # =========================
    # PROCESS OBJECTS
    # =========================
    for results, model in [
        (results1, model_coco),
        (results2, model_custom)
    ]:

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                all_boxes.append(
                    (x1, y1, x2, y2)
                )

                area = (
                    (x2 - x1) *
                    (y2 - y1)
                )

                cls = int(box.cls[0])

                name = model.names[cls]

                conf = float(box.conf[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Ignore top-half objects
                # But always allow traffic light
                if (
                    cy < h * 0.50
                    and name not in [
                        "traffic light",
                        "close door",
                        "stair",
                        "steps"
                    ]
                ):
                    continue

                # =========================
                # FILTERS
                # =========================
                if (
                    name == "wall"
                    and conf < 0.90
                ):
                    continue

                if (
                    name == "tree"
                    and conf < 0.70
                ):
                    continue

                if (
                    name == "steps"
                    and conf < 0.45
                ):
                    continue

                if (
                    name == "open manhole"
                    and conf < 0.90
                ):
                    continue

                if (
                    name == "close door"
                    and conf < 0.20
                ):
                    continue

                if (
                    name == "pothole"
                    and area < 1500
                ):
                    continue

                # Ignore small/far objects
                if (
                    area < 5000
                    and
                    name != "traffic light"
                ):
                    continue

                # =========================
                # CENTER AREA
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
    # OBJECT WARNING SYSTEM
    # =========================
    if center_objects:

        nearest = max(
            center_objects,
            key=lambda x: x["area"]
        )

        current_object = nearest["name"]

        nearest_area = nearest["area"]

        current_time = time.time()

        text = None

        # =========================
        # POTHOLE
        # =========================
        if current_object == "pothole": 
            if nearest_area > 4000: 
                direction = get_safe_direction(
                    frame, all_boxes
                ) 
                text = (
                    f"Pothole ahead. " 
                    f"{direction}" 
                )
            else: 
                text = ( 
                    "Small pothole ahead. "
                    "Walk carefully in slow speed" 
                )
        # =========================
        # STAIRS / STEPS
        # =========================
        elif current_object in [
            "stair",
            "steps"
        ]:

            stair_box = None
            largest_area = 0

            # Find biggest stair box
            for results, model in [
                (results1, model_coco),
                (results2, model_custom)
            ]:

                for r in results:

                    for box in r.boxes:

                        cls = int(box.cls[0])

                        name = model.names[cls]

                        if name not in [
                            "stair",
                            "steps"
                        ]:
                            continue

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        area = (
                            (x2 - x1)
                            *
                            (y2 - y1)
                        )

                        if area > largest_area:

                            largest_area = area

                            stair_box = (
                                x1, y1, x2, y2
                            )

            if stair_box is not None:

                x1, y1, x2, y2 = stair_box

                stair_crop = original_frame[
                    y1:y2,
                    x1:x2
                ]

                steps_count, stair_type = (
                    analyze_stairs(stair_crop)
                )

                # =========================
                # FINAL STAIR WARNING
                # =========================
                if steps_count >= 2:

                    text = (
                        f"Stairs ahead. "
                        f"Move carefully. "
                        f"{stair_type}. "
                        f"Approximately "
                        f"{steps_count} steps."
                    )

                else:

                    text = (
                        "Stairs ahead. "
                        "Move carefully."
                    )

        # =========================
        # CLOSE DOOR
        # =========================
        elif current_object == "close door":

            direction = get_safe_direction(
                frame,
                all_boxes
            )

            text = (
                "Close door ahead. " 
                "Carefully open the door then enter or " 
                f"{direction}" 
            )

        # =========================
        # BIG OBJECTS
        # =========================
        elif current_object in [
            "tree",
            "wall",
            "open manhole",
            "toilet",
            "potted plant",
            "chair",
            "bench",
            "cat",
            "dog",
            "bed",
            "dining table",
            "refrigerator"
        ]:

            direction = get_safe_direction(
                frame,
                all_boxes
            )

            text = (
                f"{current_object} ahead. "
                f"{direction}"
            )

        # =========================
        # SMART VEHICLE DETECTION
        # =========================
        elif current_object in [
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "person"
        ]:

            nearest_box = None
            nearest_area2 = 0

            # Find nearest vehicle box
            for results, model in [
                (results1, model_coco),
                (results2, model_custom)
            ]:

                for r in results:

                    for box in r.boxes:

                        cls = int(box.cls[0])
                        name = model.names[cls]

                        if name != current_object:
                            continue

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        area = (
                            (x2 - x1) *
                            (y2 - y1)
                        )

                        if area > nearest_area2:

                            nearest_area2 = area

                            nearest_box = (
                                x1, y1, x2, y2
                            )

            if nearest_box is not None:

                x1, y1, x2, y2 = nearest_box

                cx = (x1 + x2) // 2

                cy = (y1 + y2) // 2

                area = (
                    (x2 - x1) *
                    (y2 - y1)
                )

                # =========================
                # ONLY BOTTOM HALF
                # =========================
                if cy > h * 0.35:

                    # Save history
                    vehicle_id = current_object

                    vehicle_history[vehicle_id].append(
                        (cx, area)
                    )

                    text = None

                    # Need 3 frames
                    if len(
                        vehicle_history[vehicle_id]
                    ) >= 3:

                        old_x, old_area = (
                            vehicle_history[vehicle_id][0]
                        )

                        mid_x, mid_area = (
                            vehicle_history[vehicle_id][1]
                        )

                        new_x, new_area = (
                            vehicle_history[vehicle_id][2]
                        )

                        # =========================
                        # AREA GROWTH
                        # =========================
                        if old_area > 0:

                            growth_ratio = (
                                new_area / old_area
                            )

                        else:

                            growth_ratio = 1

                        # =========================
                        # SIDE MOVEMENT
                        # =========================
                        x_shift = abs(
                            new_x - old_x
                        )

                        print(
                            current_object,
                            "Growth:",
                            round(growth_ratio, 2),
                            "Shift:",
                            x_shift,
                            "Area:",
                            area
                        )

                        moving_left_to_right = (
                            (new_x - old_x)
                            > SIDE_MOVEMENT_THRESHOLD
                        )

                        moving_right_to_left = (
                            (old_x - new_x)
                            > SIDE_MOVEMENT_THRESHOLD
                        )

                        # =========================
                        # APPROACHING
                        # =========================
                        # only approaching user
                        approaching = (
                            growth_ratio > AREA_GROWTH_THRESHOLD
                            and
                            x_shift < 60
                        )
                        # =========================
                        # TTC LEVEL ESTIMATION
                        # =========================
                        if growth_ratio >= FAST_APPROACH_THRESHOLD:

                            ttc_level = "danger"

                        elif growth_ratio >= MEDIUM_APPROACH_THRESHOLD:

                            ttc_level = "warning"

                        else:

                            ttc_level = "normal"

                        # =========================
                        # VEHICLE POSITION
                        # =========================
                        if new_x < w * 0.35:

                            vehicle_side = "left"

                        elif new_x < w * 0.65:

                            vehicle_side = "center"

                        else:

                            vehicle_side = "right"

                        # =========================
                        # VEHICLE COMING TOWARD USER
                        # Straight + Diagonal
                        # =========================
                        side_crossing = (
                            x_shift > 70
                            and
                            growth_ratio < 1.20
                        )

                        if (
                            approaching
                            and
                            area > 3500
                        ):

                            direction = get_safe_direction(
                                frame,
                                all_boxes
                            )

                            # =========================
                            # VERY FAST VEHICLE
                            # =========================
                            if ttc_level == "danger":

                                text = (
                                    f"High speed "
                                    f"{current_object} "
                                    f"approaching from "
                                    f"{vehicle_side}. "
                                    f"{direction}"
                                )

                            # =========================
                            # MEDIUM SPEED
                            # =========================
                            elif ttc_level == "warning":

                                text = (
                                    f"{current_object} "
                                    f"approaching from "
                                    f"{vehicle_side}. "
                                    f"{direction}"
                                )

                            # =========================
                            # SLOW SPEED
                            # =========================
                            else:

                                text = (
                                    f"Slow "
                                    f"{current_object} "
                                    f"moving from "
                                    f"{vehicle_side}. "
                                    f"{direction}"
                                )
                        # =========================
                        # LEFT TO RIGHT
                        # =========================
                        elif moving_left_to_right:

                            text = (
                                f"{current_object} "
                                f"moving left to right"
                            )

                        # =========================
                        # RIGHT TO LEFT
                        # =========================
                        elif moving_right_to_left:

                            text = (
                                f"{current_object} "
                                f"moving right to left"
                            )

                        # =========================
                        # STOPPED / NORMAL
                        # =========================
                        else:

                            if area > 9000:

                                direction = (
                                    get_safe_direction(
                                        frame,
                                        all_boxes
                                    )
                                )

                                text = (
                                    f"{current_object} ahead. "
                                    f"{direction}"
                                )

                            else:

                                text = None

                else:

                    text = None

        elif current_object in [
            "bird",
            "bottle",
            "bowl",
            "cell phone",
            "book",
            "vase"
        ]:
        
            text = (
                f"{current_object} detected"
            )
            
        # =========================
        # UNKNOWN OBSTACLE
        # =========================
        else:

            if nearest_area > 12000:

                direction = get_safe_direction(
                    frame,
                    all_boxes
                )

                text = (
                    f"Obstacle ahead. "
                    f"{direction}"
                )

            else:

                text = None
            

        # =========================
        # SPEAK CONTROL
        # =========================
        if (
            text is not None
            and
            current_time - last_speak >= delay
        ):

            print(text)

            threading.Thread(
            target=speak,
            args=(text,),
            daemon=True
            ).start()

            last_speak = current_time

    cv2.putText(
        frame,
        f"Environment: {environment}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    current_fps_time = time.time()

    fps = 1 / (
        current_fps_time - prev_time
    )

    prev_time = current_fps_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow(
        "Smart Glass System",
        frame
    )

    # ESC EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()

cv2.destroyAllWindows()
