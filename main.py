import os
from ultralytics import YOLO
import cv2
from tracker import Tracker
import numpy as np

videoDir = os.path.\
    join('.','data','people.mp4')

cap = cv2.VideoCapture(videoDir)

tracker = Tracker()

ret, frame = cap.read()

# uses deep sort
model = YOLO("yolov8n.pt")

num_1 = 0
num_2 = 0

# variables to be used
prev_centroids = []
centroids = []
totalTime = []
prevTime = []
i2 = 0
time = 0

# loops through the frames
while ret:
    results = model(frame)
    # checks one frame at a time
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])

        # algorithm to detect people
        tracker.update(frame, detections)
        threshold = 75
        i = 0

        # adds bounding box to people
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            centroid_x = (int(y1))
            centroids.append(centroid_x)
            totalTime.append(0)
            # checks if the y values can be compared
            if prev_centroids :
                dx = centroid_x - prev_centroids[i]
                # checks if the yval is in the same range as it started
                # as long as the val is in the same area then it must not be moving that fast or at all
                if (-55 <= dx <= 55):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 100), 3)
                    prevTime[i] += 1
                    time = prevTime[i]
                    # assigns stars to sus persons
                    if(5 >= time > 1):
                        cv2.putText(frame, '*', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (0, 0, 0), 8)
                        cv2.putText(frame, '*', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (255, 255, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 125), 3)
                    elif (8 >= time > 5):
                        cv2.putText(frame, '**', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (0, 0, 0), 8)
                        cv2.putText(frame, '**', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (255, 255, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 150), 3)
                    elif (12 >= time > 8):
                        cv2.putText(frame, '***', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (0, 0, 0), 8)
                        cv2.putText(frame, '***', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (255, 255, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 175), 3)
                    elif (15 >= time > 12):
                        cv2.putText(frame, '****', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (0, 0, 0), 8)
                        cv2.putText(frame, '****', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (255, 255, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 200), 3)
                    elif (time > 15):
                        cv2.putText(frame, '*****', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (0, 0, 0), 8)
                        cv2.putText(frame, '*****', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, .27, (255, 255, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 250), 3)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 250, 0), 5)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 250), 3)
                    print(dx)
                    pass
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 0, 0), 5)
                    totalTime[i] = 0
                    # The person is moving
            else:
                # This is the first frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 0, 0), 3)
                pass
            i = i+1
        # Update the list of previous centroids
        if i2 > 0:
            print("i2: ", i2)
        else:
            prevTime = totalTime
            prev_centroids = centroids

        i2 += 1

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()