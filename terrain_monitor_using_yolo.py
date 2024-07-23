import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from twilio.rest import Client
import os
import threading

# Function for sending Twilio alerts
def send_twilio_alert(account_sid, auth_token, twilio_number, recipient_number, message):
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=recipient_number
    )
    print("Twilio alert sent.")

# Function for thread 1 (Accident detection)
def thread_1(video_path, account_sid, auth_token, twilio_number, recipient_number):
    track_history = defaultdict(lambda: [])
    model = YOLO(r"D:\downloadsssss\multithreading_test\model\accident7epochs.pt")
    names = model.model.names

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Threshold for detection confidence
    detection_threshold = 0.5

    # Counter for detected objects
    alert_count = 0

    # Variable to track if alerts were sent
    alerts_sent = False

    # Save the detected video
    detected_video_path = r"D:\downloadsssss\multithreading_test\output\detected_video1.mp4"
    result = cv2.VideoWriter(detected_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    if not result.isOpened():
        print("Error: Unable to open video writer.")
    else:
        print("Video writer opened successfully.")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, conf, track_id_tensor in zip(boxes, clss, confs, results[0].boxes.id):
                    if conf > detection_threshold:
                        # Check if predicted class is "severe" or "moderate"
                        if names[int(cls)] in ["severe", "moderate"]:
                            # Increment alert count
                            alert_count += 1

                            # Send alert via Twilio for the first three detections
                            if alert_count <= 3:
                                message = f"Object {names[int(cls)]} detected with confidence {conf:.2f}"
                                send_twilio_alert(account_sid, auth_token, twilio_number, recipient_number, message)
                                print("Alert message sent:", message)  # Print indication
                                alerts_sent = True  # Set to True if an alert is sent

                        # Store tracking history
                        track_id = int(track_id_tensor)
                        track = track_history[track_id]
                        track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                        if len(track) > 30:
                            track.pop(0)

                        # Plot tracks
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                        cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

                        annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()

    if alerts_sent:
        print("Alert messages sent.")
    else:
        print("No alerts sent.")

    if os.path.exists(detected_video_path):
        print("Detected video saved at:", detected_video_path)
    else:
        print("Error: Detected video was not saved.")

# Function for thread 2 (Animal detection)
def thread_2(video_path, account_sid, auth_token, twilio_number, recipient_number):
    track_history = defaultdict(lambda: [])
    model = YOLO(r"D:\downloadsssss\multithreading_test\model\animal.pt")
    names = model.model.names

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Threshold for detection confidence
    detection_threshold = 0.5

    # Counter for detected objects
    alert_count = 0

    # Variable to track if alerts were sent
    alerts_sent = False

    # Save the detected video
    detected_video_path =r"D:\downloadsssss\multithreading_test\output\detected_video2.mp4"
    result = cv2.VideoWriter(detected_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    if not result.isOpened():
        print("Error: Unable to open video writer.")
    else:
        print("Video writer opened successfully.")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, conf, track_id_tensor in zip(boxes, clss, confs, results[0].boxes.id):
                    if conf > detection_threshold:
                        # Check if predicted class is "ELEPHANT" or "TIGER"
                        if names[int(cls)] in ["ELEPHANT", "TIGER"]:
                            # Increment alert count
                            alert_count += 1

                            # Send alert via Twilio for the first three detections
                            if alert_count <= 3:
                                message = f"Object {names[int(cls)]} detected with confidence {conf:.2f}"
                                send_twilio_alert(account_sid, auth_token, twilio_number, recipient_number, message)
                                print("Alert message sent:", message)  # Print indication
                                alerts_sent = True  # Set to True if an alert is sent

                        # Store tracking history
                        track_id = int(track_id_tensor)
                        track = track_history[track_id]
                        track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                        if len(track) > 30:
                            track.pop(0)

                        # Plot tracks
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                        cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

                        annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()

    if alerts_sent:
        print("Alert messages sent.")
    else:
        print("No alerts sent.")

    if os.path.exists(detected_video_path):
        print("Detected video saved at:", detected_video_path)
    else:
        print("Error: Detected video was not saved.")

# Function for thread 3 (Flood detection)
def thread_3(video_path, account_sid, auth_token, twilio_number, recipient_number):
    track_history = defaultdict(lambda: [])
    model = YOLO(r"D:\downloadsssss\multithreading_test\model\Flood.pt")  # segmentation model
    names = model.model.names
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(r"D:\downloadsssss\multithreading_test\output\flood_detection.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    alert_sent = 0

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.predict(im0)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                if names[int(cls)] == 'flooding':
                    if alert_sent < 3:
                        message = "Flooding detected!"
                        send_twilio_alert(account_sid, auth_token, twilio_number, recipient_number, message)
                        alert_sent += 1
                    # Draw a rectangle around the detected flooding region
                    x, y, w, h = cv2.boundingRect(mask)
                    cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(im0, 'Flooding', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(im0)

    out.release()
    cap.release()





# Input video path
video_path = r"D:\downloadsssss\multithreading_test\input\accv3.mp4"

# Twilio credentials for thread 1
thread_1_account_sid = ''
thread_1_auth_token = ''
thread_1_twilio_number = ''
thread_1_recipient_number = ''

# Twilio credentials for thread 2
thread_2_account_sid = ''
thread_2_auth_token = ''
thread_2_twilio_number = ''
thread_2_recipient_number = ''

# Twilio credentials for thread 3
thread_3_account_sid = ""
thread_3_auth_token = ''
thread_3_twilio_number = ""
thread_3_recipient_number = ""

# Create threads
t1 = threading.Thread(target=thread_1, args=(video_path, thread_1_account_sid, thread_1_auth_token, thread_1_twilio_number, thread_1_recipient_number))
t2 = threading.Thread(target=thread_2, args=(video_path, thread_2_account_sid, thread_2_auth_token, thread_2_twilio_number, thread_2_recipient_number))
t3 = threading.Thread(target=thread_3, args=(video_path, thread_3_account_sid, thread_3_auth_token, thread_3_twilio_number, thread_3_recipient_number))

# Start threads
t1.start()
t2.start()
t3.start()

# Wait for threads to finish
t1.join()
t2.join()
t3.join()

print("All threads have finished executing.")
