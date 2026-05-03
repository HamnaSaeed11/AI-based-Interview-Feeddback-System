### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
from time import sleep
import re
import os
import requests
import argparse
from collections import OrderedDict

### Image processing ###
import cv2
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import dlib
from imutils import face_utils

### Model ###
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


# ── Confidence threshold: predictions below this are treated as unreliable ──
# dlib's face detector returns a confidence score; faces detected below this
# are likely partial faces, masked faces, or false positives.
FACE_CONFIDENCE_THRESHOLD = 0.5

# Minimum face size in pixels — faces smaller than this are too small to
# give reliable emotion predictions (far away, partially hidden, etc.)
MIN_FACE_SIZE = 60

# How many consecutive frames without a face before we show the "no face" warning
NO_FACE_PATIENCE = 8

# Brightness threshold: mean pixel value below this = frame is too dark
DARK_FRAME_THRESHOLD = 25


def _is_dark_frame(gray):
    """Return True if the frame is too dark to be useful."""
    return float(np.mean(gray)) < DARK_FRAME_THRESHOLD


def _overlay_warning(frame, message, second_line=None, color=(0, 60, 220)):
    """
    Draw a semi-transparent warning banner at the centre-bottom of the frame.
    Two lines supported.
    """
    h, w = frame.shape[:2]
    banner_h = 70 if second_line else 48
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness  = 2

    # Centre first line
    (tw, th), _ = cv2.getTextSize(message, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = h - banner_h + th + 8
    cv2.putText(frame, message, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    if second_line:
        (tw2, th2), _ = cv2.getTextSize(second_line, font, font_scale - 0.1, thickness - 1)
        tx2 = (w - tw2) // 2
        ty2 = ty + th2 + 10
        cv2.putText(frame, second_line, (tx2, ty2), font, font_scale - 0.1,
                    (230, 230, 230), thickness - 1)


def _draw_emotion_report(frame, prediction, face_index, face_rect):
    """Draw the emotion probabilities panel and emotion label for one face."""
    x, y, w, h = face_rect
    emotions    = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    colors_bgr  = [
        (0,   0,   220),  # Angry   – red
        (0,  140,   0),   # Disgust – dark green
        (180,  0, 180),   # Fear    – purple
        (0,  200,   0),   # Happy   – bright green
        (220, 130,   0),  # Sad     – orange-blue
        (0,  200, 200),   # Surprise– cyan
        (160, 160, 160),  # Neutral – grey
    ]

    best_idx  = int(np.argmax(prediction[0]))
    best_prob = float(prediction[0][best_idx])

    # Bounding box — colour reflects dominant emotion
    box_color = colors_bgr[best_idx]
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Face label above the box
    label = f"Face #{face_index + 1}  {emotions[best_idx]} ({int(best_prob*100)}%)"
    cv2.putText(frame, label, (x, max(y - 10, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # Sidebar emotion report
    panel_x = 10
    panel_y = 90 + face_index * 190
    cv2.putText(frame, f"-- Face #{face_index+1} report --",
                (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    for ei, (emo, prob) in enumerate(zip(emotions, prediction[0])):
        txt   = f"{emo}: {prob:.2f}"
        color = colors_bgr[ei] if ei == best_idx else (180, 180, 180)
        cv2.putText(frame, txt,
                    (panel_x, panel_y + 18 + ei * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def gen(max_time=15):
    """
    Video streaming generator function.
    Yields MJPEG frames with emotion overlay, face-not-detected warnings,
    mask/low-confidence warnings, and dark-frame warnings.
    """

    video_capture = cv2.VideoCapture(0)

    shape_x = shape_y = 48

    # Load model and detectors once
    model               = load_model('Models/video.h5')
    face_detect         = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")

    (lStart, lEnd)   = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd)   = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (nStart, nEnd)   = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd)   = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd)   = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    predictions = []
    angry_0  = []; disgust_1 = []; fear_2   = []
    happy_3  = []; sad_4     = []; surprise_5= []; neutral_6 = []

    no_face_frames   = 0   # consecutive frames with zero usable faces
    start = time.time()
    end   = 0

    while end - start < max_time:
        end = time.time()

        ret, frame = video_capture.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Dark-frame / camera-covered check ─────────────────────────────────
        if _is_dark_frame(gray):
            _overlay_warning(
                frame,
                "⚠  Camera is too dark or covered",
                "Please turn on the lights or uncover the camera",
                color=(30, 30, 30)
            )
            cv2.putText(frame, f"Time: {int(max_time - (end - start))}s remaining",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imwrite('tmp/t.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + open('tmp/t.jpg', 'rb').read() + b'\r\n')
            continue

        # ── Face detection (with confidence scores) ───────────────────────────
        # dlib's detect() returns (rects, scores, idx); scores are confidence values
        rects, scores, _ = face_detect.run(gray, 1, -0.5)

        usable_faces = []
        low_conf_faces = []

        for rect, score in zip(rects, scores):
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # Skip faces that are too small (far away / partially in frame)
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            if score < FACE_CONFIDENCE_THRESHOLD:
                low_conf_faces.append((rect, score, (x, y, w, h)))
            else:
                usable_faces.append((rect, score, (x, y, w, h)))

        # ── No face detected ──────────────────────────────────────────────────
        if len(usable_faces) == 0 and len(low_conf_faces) == 0:
            no_face_frames += 1
            cv2.putText(frame, f"Faces: 0  |  Time: {int(max_time - (end - start))}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            if no_face_frames >= NO_FACE_PATIENCE:
                _overlay_warning(
                    frame,
                    "⚠  Face not visible  —  please face the camera",
                    "Make sure your face is well-lit and centred",
                    color=(0, 40, 160)
                )
        else:
            no_face_frames = 0  # reset counter once a face is seen again

        # ── Low-confidence faces (mask, partial, obscured) ────────────────────
        for rect, score, (x, y, w, h) in low_conf_faces:
            # Draw an orange dashed-style rectangle to mark the uncertain face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(frame,
                        f"Face unclear ({int(score*100)}% conf) — remove mask?",
                        (x, max(y - 10, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if low_conf_faces and len(usable_faces) == 0:
            _overlay_warning(
                frame,
                "⚠  Face detected but unclear  —  is your face covered?",
                "Remove mask / glasses or improve lighting for accurate analysis",
                color=(0, 80, 180)
            )

        # ── Emotion prediction on usable faces ───────────────────────────────
        for face_index, (rect, score, (x, y, w, h)) in enumerate(usable_faces):
            face = gray[y:y + h, x:x + w]
            if face.size == 0:
                continue

            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Resize and normalise
            try:
                face_resized = zoom(face, (shape_x / face.shape[0],
                                           shape_y / face.shape[1]))
            except Exception:
                continue

            face_resized = face_resized.astype(np.float32)
            face_max = float(face_resized.max())
            if face_max == 0:
                continue
            face_resized /= face_max
            face_input = np.reshape(face_resized.flatten(), (1, 48, 48, 1))

            prediction = model.predict(face_input, verbose=0)

            angry_0.append(float(prediction[0][0]))
            disgust_1.append(float(prediction[0][1]))
            fear_2.append(float(prediction[0][2]))
            happy_3.append(float(prediction[0][3]))
            sad_4.append(float(prediction[0][4]))
            surprise_5.append(float(prediction[0][5]))
            neutral_6.append(float(prediction[0][6]))

            predictions.append(str(int(np.argmax(prediction))))

            # Draw emotion overlay
            _draw_emotion_report(frame, prediction, face_index, (x, y, w, h))

            # Landmarks
            for (px, py) in shape:
                cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)

            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            for hull_pts in [leftEye, rightEye,
                              shape[nStart:nEnd], shape[mStart:mEnd],
                              shape[jStart:jEnd],
                              shape[eblStart:eblEnd], shape[ebrStart:ebrEnd]]:
                cv2.drawContours(frame, [cv2.convexHull(hull_pts)], -1, (0, 255, 0), 1)

        # ── HUD: face count + timer ────────────────────────────────────────────
        n_total   = len(usable_faces) + len(low_conf_faces)
        time_left = max(0, int(max_time - (end - start)))
        cv2.putText(frame,
                    f"Faces: {n_total}  |  Time: {time_left}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Write and yield frame
        cv2.imwrite('tmp/t.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + open('tmp/t.jpg', 'rb').read() + b'\r\n')

        # ── Save results near the end ─────────────────────────────────────────
        if end - start > max_time - 1:
            if predictions:
                with open("static/js/db/histo_perso.txt", "w") as d:
                    d.write("density\n")
                    for val in predictions:
                        d.write(str(val) + '\n')

                with open("static/js/db/histo.txt", "a") as d:
                    for val in predictions:
                        d.write(str(val) + '\n')

                import csv
                rows = list(zip(angry_0, disgust_1, fear_2,
                                happy_3, sad_4, surprise_5, neutral_6))
                with open("static/js/db/prob.csv", "w") as d:
                    csv.writer(d).writerows(rows)
                with open("static/js/db/prob_tot.csv", "a") as d:
                    csv.writer(d).writerows(rows)

            K.clear_session()
            break

    video_capture.release()
