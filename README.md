# Multithreaded Object Detection and Instance Segmentation With Object Tracking System with Twilio Alerts Using Yolo

This project uses YOLO (You Only Look Once) models to detect various events (accidents, animals, floods) in a video file and sends alerts via Twilio. The system runs each detection model in a separate thread for efficient processing.

## Requirements

- Python 3.7 or above
- OpenCV
- NumPy
- Ultralytics YOLO
- Twilio

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multithreading-detection.git
    cd multithreading-detection
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python-headless numpy ultralytics twilio
    ```

## Usage

1. **Download the YOLO models**:
   - Ensure you have the required YOLO models for accident detection (`accident7epochs.pt`), animal detection (`animal.pt`), and flood detection (`Flood.pt`). Place them in the appropriate directory (e.g., `model` folder).

2. **Set up Twilio**:
   - Sign up for a Twilio account and get your Account SID, Auth Token, Twilio Number, and Recipient Number.
   - Update the Twilio credentials in the script:

     ```python
     # Twilio credentials for thread 1
     thread_1_account_sid = 'your_account_sid'
     thread_1_auth_token = 'your_auth_token'
     thread_1_twilio_number = 'your_twilio_number'
     thread_1_recipient_number = 'recipient_number'

     # Twilio credentials for thread 2
     thread_2_account_sid = 'your_account_sid'
     thread_2_auth_token = 'your_auth_token'
     thread_2_twilio_number = 'your_twilio_number'
     thread_2_recipient_number = 'recipient_number'

     # Twilio credentials for thread 3
     thread_3_account_sid = 'your_account_sid'
     thread_3_auth_token = 'your_auth_token'
     thread_3_twilio_number = 'your_twilio_number'
     thread_3_recipient_number = 'recipient_number'
     ```

3. **Run the script**:
   - Set the path to the input video file and execute the script.

     ```python
     # Input video path
     video_path = r"path_to_your_video.mp4"

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
     ```

4. **Output**:
   - The detected videos will be saved in the `output` directory.
   - Alerts will be sent to the specified recipient via Twilio.

## Project Structure

```
multithreading-detection/
│
├── model/
│   ├── accident.pt
│   ├── animal.pt
│   └── Flood.pt
│
├── input/
│   └── your_video.mp4
│
├── output/
│   ├── acc_detection_video1.mp4
│   ├── wild_animal_detection_video2.mp4
│   └── flood_detection_video3
.mp4
│
├── README.md
└── main.py
```

## Important Notes

- Ensure that the paths to the YOLO models and the input video are correct.
- The detection threshold and alert count can be adjusted based on the requirements.
- Ensure that the Twilio credentials are securely stored and not hard-coded in production environments.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

