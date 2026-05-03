### General imports ###

from __future__ import division
from ffmpeg_path import *
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt
import subprocess
import io

### Flask imports
from flask import Flask, render_template, session, request, redirect, flash, Response
from functools import wraps
import hashlib

### Audio imports ###
from library.speech_emotion_recognition import *

### Video imports ###
from library.video_emotion_recognition import *

### Text imports ###
from library.text_emotion_recognition import *
from library.text_preprocessor import *
from nltk import *
from tika import parser
from werkzeug.utils import secure_filename
import tempfile

# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = 'tmp'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB


################################################################################
################################## AUTH ########################################
################################################################################

USERS_FILE = os.path.join('static', 'js', 'db', 'users.txt')


def _hash(password):
    """SHA-256 hash of password — no plaintext stored."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def _load_users():
    """Return dict of {username: hashed_password} from users.txt."""
    users = {}
    if not os.path.exists(USERS_FILE):
        return users
    with open(USERS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                username, hashed = line.split('|', 1)
                users[username.strip()] = hashed.strip()
    return users


def _save_user(username, hashed_password):
    """Append a new user to users.txt."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'a') as f:
        f.write(f"{username}|{hashed_password}\n")


def login_required(f):
    """Decorator — redirects to /login if user is not in session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect('/')
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        users = _load_users()
        if username not in users:
            flash('Account not found. Please sign up first.', 'error')
            return render_template('login.html')
        if users[username] != _hash(password):
            flash('Incorrect password. Please try again.', 'error')
            return render_template('login.html')
        session['username'] = username
        return redirect('/')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'username' in session:
        return redirect('/')
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm',  '').strip()
        # Validate
        import re as _re
        if not username or not password or not confirm:
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        if not _re.match(r'^[A-Za-z0-9_]+$', username):
            flash('Username may only contain letters, numbers, and underscores.', 'error')
            return render_template('signup.html')
        if len(password) < 4:
            flash('Password must be at least 4 characters.', 'error')
            return render_template('signup.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        users = _load_users()
        if username in users:
            flash('That username is already taken. Please choose another.', 'error')
            return render_template('signup.html')
        _save_user(username, _hash(password))
        flash('Account created! You can now log in.', 'success')
        return redirect('/login')
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect('/login')

@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template('index.html', username=session.get('username', ''))


################################################################################
################################## RULES #######################################
################################################################################

@app.route('/rules')
@login_required
def rules():
    return render_template('rules.html')


################################################################################
############################### VIDEO INTERVIEW ################################
################################################################################

df = pd.read_csv('static/js/db/histo.txt', sep=",")

@app.route('/video', methods=['GET', 'POST'])
@login_required
def video():
    flash('You will have 45 seconds to discuss the topic mentioned above. Due to restrictions, we are not able to redirect you once the video is over. Please move your URL to /video_dash instead of /video_1 once over. You will be able to see your results then.')
    return render_template('video.html')

@app.route('/video_1', methods=['POST'])
@login_required
def video_1():
    try:
        duration = int(request.form.get('duration', 15))
        duration = max(10, min(120, duration))
    except (ValueError, TypeError):
        duration = 15
    session['rec_duration'] = duration
    return render_template('video_1.html', duration=duration)

@app.route('/video_stream', methods=['GET'])
@login_required
def video_stream():
    try:
        duration = int(session.get('rec_duration', 15))
        duration = max(10, min(120, duration))
    except (ValueError, TypeError):
        duration = 15
    try:
        return Response(gen(max_time=duration), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        return None

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'video_file' not in request.files:
        flash('No file selected')
        return redirect('/video')
    file = request.files['video_file']
    if file.filename == '':
        flash('No file selected')
        return redirect('/video')
    if file and allowed_video_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        process_uploaded_video(filepath)
        return redirect('/video_dash')
    else:
        flash('Invalid file format. Please upload MP4, AVI, or MOV files.')
        return redirect('/video')

def process_uploaded_video(filepath):
    import cv2, csv
    from tensorflow.keras.models import load_model
    import dlib
    from imutils import face_utils

    shape_x = shape_y = 48
    model = load_model('Models/video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")
    video_capture = cv2.VideoCapture(filepath)
    if not video_capture.isOpened():
        print("Error opening video file")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = 5
    current_frame = 0
    predictions = []
    prob_cols = [[], [], [], [], [], [], []]
    print(f"Processing video: {total_frames} frames at {fps} FPS")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if current_frame % frame_step != 0:
            current_frame += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h, x:x+w]
            if face.size == 0:
                continue
            try:
                face = cv2.resize(face, (shape_x, shape_y))
            except:
                continue
            face = face.astype(np.float32)
            face /= float(face.max())
            face = np.reshape(face, (1, shape_x, shape_y, 1))
            prediction = model.predict(face)
            for col_idx in range(7):
                prob_cols[col_idx].append(prediction[0][col_idx].astype(float))
            predictions.append(str(np.argmax(prediction)))
        current_frame += 1
        if current_frame > 300:
            break

    video_capture.release()

    if predictions:
        with open("static/js/db/histo_perso.txt", "w") as d:
            d.write("density\n")
            for val in predictions:
                d.write(str(val) + '\n')
        with open("static/js/db/histo.txt", "a") as d:
            for val in predictions:
                d.write(str(val) + '\n')
        rows = zip(*prob_cols)
        with open("static/js/db/prob.csv", "w") as d:
            csv.writer(d).writerows(rows)
        rows2 = zip(*prob_cols)
        with open("static/js/db/prob_tot.csv", "a") as d:
            csv.writer(d).writerows(rows2)
    print(f"Processed {len(predictions)} frames")

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

@app.route('/video_dash', methods=("POST", "GET"))
def video_dash():
    try:
        df_2 = pd.read_csv('static/js/db/histo_perso.txt')
    except FileNotFoundError:
        df_2 = pd.DataFrame(columns=['density'])

    def emo_prop(df_2):
        if df_2.empty:
            return [0, 0, 0, 0, 0, 0, 0]
        n = len(df_2)
        return [int(100 * len(df_2[df_2.density == i]) / n) for i in range(7)]

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    emo_perso = {e: (len(df_2[df_2.density == i]) if not df_2.empty else 0) for i, e in enumerate(emotions)}
    emo_glob  = {e: (len(df[df.density == i]) if not df.empty else 0) for i, e in enumerate(emotions)}

    pd.DataFrame(list(emo_perso.items()), columns=['EMOTION', 'VALUE']).to_csv('static/js/db/hist_vid_perso.txt', sep=",", index=False)
    pd.DataFrame(list(emo_glob.items()),  columns=['EMOTION', 'VALUE']).to_csv('static/js/db/hist_vid_glob.txt',  sep=",", index=False)

    emotion       = df_2.density.mode()[0] if not df_2.empty else 0
    emotion_other = df.density.mode()[0]   if not df.empty  else 0

    def emotion_label(e):
        return ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"][e] if 0 <= e <= 6 else "Neutral"

    try:
        df_altair = pd.read_csv('static/js/db/prob.csv', header=None).reset_index()
        df_altair.columns = ['Time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    except Exception:
        df_altair = pd.DataFrame({'Time':[0],'Angry':[0],'Disgust':[0],'Fear':[0],'Happy':[0],'Sad':[0],'Surprise':[0],'Neutral':[0]})

    angry_c   = alt.Chart(df_altair).mark_line(color='orange',  strokeWidth=2).encode(x='Time:Q', y='Angry:Q',    tooltip=["Angry"])
    disgust_c = alt.Chart(df_altair).mark_line(color='red',     strokeWidth=2).encode(x='Time:Q', y='Disgust:Q',  tooltip=["Disgust"])
    fear_c    = alt.Chart(df_altair).mark_line(color='green',   strokeWidth=2).encode(x='Time:Q', y='Fear:Q',     tooltip=["Fear"])
    happy_c   = alt.Chart(df_altair).mark_line(color='blue',    strokeWidth=2).encode(x='Time:Q', y='Happy:Q',    tooltip=["Happy"])
    sad_c     = alt.Chart(df_altair).mark_line(color='black',   strokeWidth=2).encode(x='Time:Q', y='Sad:Q',      tooltip=["Sad"])
    surprise_c= alt.Chart(df_altair).mark_line(color='pink',    strokeWidth=2).encode(x='Time:Q', y='Surprise:Q', tooltip=["Surprise"])
    neutral_c = alt.Chart(df_altair).mark_line(color='brown',   strokeWidth=2).encode(x='Time:Q', y='Neutral:Q',  tooltip=["Neutral"])
    chart = (angry_c + disgust_c + fear_c + happy_c + sad_c + surprise_c + neutral_c).properties(width=1000, height=400, title='Probability of each emotion over time')
    chart.save('static/CSS/chart.html')

    prob_list = emo_prop(df_2)
    happy_pct   = prob_list[3]; neutral_pct = prob_list[6]
    fear_pct    = prob_list[2]; angry_pct   = prob_list[0]
    disgust_pct = prob_list[1]; sad_pct     = prob_list[4]

    _score = 0.0
    if happy_pct >= 45:   _score += 45
    elif happy_pct >= 38: _score += 40
    elif happy_pct >= 30: _score += 32
    elif happy_pct >= 22: _score += 22
    elif happy_pct >= 15: _score += 12
    elif happy_pct >= 8:  _score += 5

    if 15 <= neutral_pct <= 38:   _score += 20
    elif 10 <= neutral_pct < 15:  _score += 16
    elif 38 < neutral_pct <= 48:  _score += 12
    elif 48 < neutral_pct <= 58:  _score += 5
    elif neutral_pct < 10:        _score += 10

    if fear_pct <= 5:    _score += 18
    elif fear_pct <= 10: _score += 14
    elif fear_pct <= 15: _score += 8
    elif fear_pct <= 22: _score += 3

    _neg = angry_pct + disgust_pct
    if _neg <= 4:    _score += 12
    elif _neg <= 8:  _score += 9
    elif _neg <= 14: _score += 5
    elif _neg <= 20: _score += 2

    if sad_pct <= 6:    _score += 5
    elif sad_pct <= 12: _score += 3
    elif sad_pct <= 20: _score += 1

    _score -= 3
    raw_score = max(0, min(100, int(_score)))
    score_label = "Excellent" if raw_score >= 75 else "Good" if raw_score >= 55 else "Average" if raw_score >= 38 else "Needs Work"
    score_color = "#28a745"   if raw_score >= 75 else "#45AF7F" if raw_score >= 55 else "#ff9800" if raw_score >= 38 else "#e53935"

    def generate_suggestions(p):
        s = {'strengths': [], 'areas_for_improvement': [], 'specific_tips': []}
        if p[3] > 30: s['strengths'].append("You show good confidence and positive energy")
        else:
            s['areas_for_improvement'].append("Try to express more happiness/confidence")
            s['specific_tips'].append("Practice natural smiling and positive body language")
        if p[2] > 20:
            s['areas_for_improvement'].append("High anxiety levels detected")
            s['specific_tips'].append("Prepare thoroughly and practice relaxation techniques")
        if p[0] > 15:
            s['areas_for_improvement'].append("Avoid showing anger/frustration")
            s['specific_tips'].append("Stay calm and composed throughout the interview")
        if p[6] > 50:
            s['areas_for_improvement'].append("Show more expressive emotions")
            s['specific_tips'].append("Vary your facial expressions to appear more engaged")
        s['specific_tips'].extend([
            "Maintain good eye contact with the camera",
            "Keep consistent lighting on your face",
            "Speak clearly and at a moderate pace",
            "Use natural hand gestures to emphasize points"
        ])
        return s

    return render_template('video_dash.html',
                           emo=emotion_label(emotion),
                           prob=emo_prop(df_2),
                           suggestions=generate_suggestions(prob_list),
                           score=raw_score, score_label=score_label, score_color=score_color)


################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

@app.route('/audio_index', methods=['POST'])
@login_required
def audio_index():
    flash("After pressing the button above, you will have 15sec to answer the question.")
    return render_template('audio.html', display_button=False)

@app.route('/audio_recording', methods=("POST", "GET"))
@login_required
def audio_recording():
    SER = speechEmotionRecognition()
    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=16)
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions.")
    return render_template('audio.html', display_button=True)

@app.route('/audio_dash', methods=("POST", "GET"))
@login_required
def audio_dash():
    model_sub_dir = os.path.join('Models', 'audio.hdf5')
    SER = speechEmotionRecognition(model_sub_dir)
    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')

    step = 1
    sample_rate = 16000
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step * sample_rate)

    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    major_emotion = max(set(emotions), key=emotions.count) if emotions else "Neutral"
    emotion_dist  = [int(100 * emotions.count(e) / len(emotions)) if emotions else 0 for e in SER._emotion.values()]

    df_dist = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_dist.to_csv(os.path.join('static/js/db', 'audio_emotions_dist.txt'), sep=',')

    try:
        df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")
        major_emotion_other  = df_other.EMOTION.mode()[0] if not df_other.empty else "Neutral"
        emotion_dist_other   = [int(100 * len(df_other[df_other.EMOTION == e]) / len(df_other)) if not df_other.empty else 0 for e in SER._emotion.values()]
    except FileNotFoundError:
        major_emotion_other  = "Neutral"
        emotion_dist_other   = [0] * 7

    df_other_out = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other_out.to_csv(os.path.join('static/js/db', 'audio_emotions_dist_other.txt'), sep=',')

    time.sleep(0.5)
    return render_template('audio_dash.html', emo=major_emotion, emo_other=major_emotion_other,
                           prob=emotion_dist, prob_other=emotion_dist_other)


# ── Audio upload (from browser MediaRecorder or manual file upload) ──

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'ogg', 'm4a', 'flac', 'webm'}

@app.route('/upload_audio', methods=['POST'])
@login_required
def upload_audio():
    if 'audio_file' not in request.files:
        flash('No audio file selected.')
        return redirect('/')

    f = request.files['audio_file']
    if f.filename == '':
        flash('No audio file selected.')
        return redirect('/')

    if not allowed_audio_file(f.filename):
        flash('Unsupported format. Please upload WAV, MP3, OGG, M4A, FLAC, or WEBM.')
        return redirect('/')

    raw_bytes = f.read()

    # Detect real format from magic bytes — browser MediaRecorder sends webm
    # even when the filename says .wav
    if raw_bytes[:4] == b'\x1a\x45\xdf\xa3':
        actual_ext = 'webm'
    elif raw_bytes[:4] == b'OggS':
        actual_ext = 'ogg'
    elif raw_bytes[:3] in (b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
        actual_ext = 'mp3'
    else:
        actual_ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else 'wav'

    rec_sub_dir = os.path.join('tmp', 'voice_recording.wav')

    if actual_ext == 'wav':
        with open(rec_sub_dir, 'wb') as out:
            out.write(raw_bytes)
    else:
        # Save raw input then convert to 16 kHz mono WAV via ffmpeg
        tmp_input = os.path.join('tmp', 'upload_input.' + actual_ext)
        with open(tmp_input, 'wb') as out:
            out.write(raw_bytes)
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', tmp_input,
                 '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', rec_sub_dir],
                capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode())
        except Exception as ffmpeg_err:
            print(f"[upload_audio ffmpeg error] {ffmpeg_err}")
            # Fallback: pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(io.BytesIO(raw_bytes), format=actual_ext)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(rec_sub_dir, format='wav')
            except Exception as pydub_err:
                print(f"[upload_audio pydub error] {pydub_err}")
                flash('Could not convert audio. Install ffmpeg or upload a WAV file directly.')
                return redirect('/')

    return redirect('/audio_dash')


################################################################################
############################### TEXT INTERVIEW #################################
################################################################################

tempdirectory = tempfile.gettempdir()

# TEXT ROUTES DISABLED - hidden from frontend
# @app.route('/text', methods=['POST'])
# def text():
#     return render_template('text.html')

def get_personality(text):
    """
    Returns (predictions_array, None) on success, or (None, error_message) on failure.
    Always prints the real traceback so it is visible in the Flask console.
    """
    import traceback
    try:
        pred = predict().run(text, model_name="Personality_traits_NN")
        if pred is None:
            return None, "Model returned None — check that Models/Personality_traits_NN.json and .h5 are present and compatible."
        return pred, None
    except FileNotFoundError as e:
        msg = f"Model file not found: {e}"
        print(f"[get_personality ERROR] {msg}")
        traceback.print_exc()
        return None, msg
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"[get_personality ERROR] {msg}")
        traceback.print_exc()
        return None, msg

def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts

def preprocess_text(text):
    return NLTKPreprocessor().transform([text])

def _save_text_data(probas_raw, traits, text_input):
    """Shared logic for text_1 and text_pdf: saves all CSV files and returns render args."""
    # Write floats (0-1) to text_perso.txt — D3 reads these directly
    df_text_perso = pd.DataFrame(list(zip(traits, probas_raw)), columns=['Trait', 'Value'])
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)

    # Accumulate history — guard against corrupt files (e.g. git conflict markers)
    try:
        df_text = pd.read_csv('static/js/db/text.txt', sep=",")
        if not all(c in df_text.columns for c in traits):
            df_text = pd.DataFrame(columns=traits)
        # Check for non-numeric data (git conflict markers etc.)
        for t in traits:
            pd.to_numeric(df_text[t], errors='raise')
    except Exception:
        df_text = pd.DataFrame(columns=traits)

    df_new = pd.concat([df_text, pd.DataFrame([probas_raw], columns=traits)], ignore_index=True)
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)

    # Mean across all candidates (floats 0-1 for D3)
    means = {t: float(np.mean(df_new[t])) for t in traits}
    df_mean = pd.DataFrame(list(means.items()), columns=['Trait', 'Value'])
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.loc[df_mean['Value'].idxmax(), 'Trait']

    # Percentages for HTML display
    probas_pct        = [int(v * 100) for v in probas_raw]
    probas_others_pct = [int(means[t] * 100) for t in traits]
    trait             = traits[probas_pct.index(max(probas_pct))]

    # Word analysis
    try:
        preprocessed = preprocess_text(text_input)
        _, num_words, counts = get_text_info(preprocessed)
    except Exception:
        num_words, counts = len(text_input.split()), {}

    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ\n")
        for word, freq in counts.items():
            d.write(f"{word},{freq}\n")

    with open("static/js/db/words_common.txt", "a") as d:
        for word, freq in counts.items():
            d.write(f"{word},{freq}\n")

    try:
        df_wc = pd.read_csv('static/js/db/words_common.txt', sep=',', on_bad_lines='skip')
        df_wc['FREQ'] = pd.to_numeric(df_wc['FREQ'], errors='coerce')
        df_wc = df_wc.groupby('WORDS').sum().reset_index()
        df_wc.to_csv('static/js/db/words_common.txt', sep=",", index=False)
        common_words_others = df_wc.sort_values('FREQ', ascending=False)['WORDS'][:15].tolist()
    except Exception:
        common_words_others = []

    try:
        df_wp = pd.read_csv('static/js/db/words_perso.txt', sep=',', on_bad_lines='skip')
        common_words_perso = df_wp.sort_values('FREQ', ascending=False)['WORDS'][:15].tolist()
    except Exception:
        common_words_perso = []

    return dict(traits=probas_pct, trait=trait, trait_others=trait_others,
                probas_others=probas_others_pct, num_words=num_words,
                common_words=common_words_perso, common_words_others=common_words_others)


# @app.route('/text_1', methods=['POST'])
# def text_1():
    text_input = request.form.get('text', '').strip()
    if not text_input:
        flash('Please enter some text before submitting.')
        return render_template('text.html')

    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

    result, err = get_personality(text_input)
    if result is not None:
        probas_raw = [float(v) for v in result[0].tolist()]
    else:
        flash(f'Personality model error — {err}')
        probas_raw = [0.2, 0.2, 0.2, 0.2, 0.2]

    render_args = _save_text_data(probas_raw, traits, text_input)
    return render_template('text_dash.html', **render_args)


ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath, ext):
    """
    Extract plain text from PDF, DOCX, DOC, or TXT.
    Returns (text_string, error_message_or_None).
    """
    ext = ext.lower()

    if ext == 'txt':
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                return f.read().strip(), None
        except Exception as e:
            return None, f"Could not read text file: {e}"

    if ext == 'pdf':
        # Try pypdf first (no Java required)
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            pages = [page.extract_text() or '' for page in reader.pages]
            text = '\n'.join(pages).strip()
            if text:
                return text, None
        except ImportError:
            pass
        except Exception as e:
            print(f"[extract_text pypdf error] {e}")
        # Fallback: Apache Tika (requires Java)
        try:
            from tika import parser as tika_parser
            content = (tika_parser.from_file(filepath).get('content') or '').strip()
            if content:
                return content, None
            return None, "PDF text extraction returned empty content. Is the file scanned/image-only?"
        except Exception as e:
            print(f"[extract_text tika error] {e}")
            return None, f"PDF extraction failed: {e}. Install Java (required by Tika) or use pypdf."

    if ext in ('docx', 'doc'):
        # python-docx handles .docx natively
        try:
            import docx as python_docx
            doc = python_docx.Document(filepath)
            text = '\n'.join(p.text for p in doc.paragraphs).strip()
            if text:
                return text, None
        except ImportError:
            pass
        except Exception as e:
            print(f"[extract_text python-docx error] {e}")
        # Fallback: Tika (also supports .doc)
        try:
            from tika import parser as tika_parser
            content = (tika_parser.from_file(filepath).get('content') or '').strip()
            if content:
                return content, None
            return None, "Could not extract text from this Word document."
        except Exception as e:
            return None, f"Word extraction failed: {e}. Try installing python-docx."

    return None, f"Unsupported file type: {ext}"


# @app.route('/text_pdf', methods=['POST'])
# def text_pdf():
    f = request.files.get('file')
    if not f or f.filename == '':
        flash('No file selected. Please choose a PDF, DOCX, DOC, or TXT file.')
        return redirect('/text')

    if not allowed_file(f.filename):
        flash('Unsupported file type. Please upload a PDF, DOCX, DOC, or TXT file.')
        return redirect('/text')

    filename = secure_filename(f.filename)
    ext = filename.rsplit('.', 1)[-1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    f.save(filepath)

    text_input, err = extract_text_from_file(filepath, ext)
    if not text_input:
        flash(f'Could not extract text: {err}')
        return redirect('/text')

    if len(text_input.split()) < 10:
        flash('The extracted text is too short for a reliable personality analysis (less than 10 words). '
              'Please upload a longer document.')
        return redirect('/text')

    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

    result, err = get_personality(text_input)
    if result is not None:
        probas_raw = [float(v) for v in result[0].tolist()]
    else:
        flash(f'Personality model error — {err}')
        probas_raw = [0.2, 0.2, 0.2, 0.2, 0.2]

    render_args = _save_text_data(probas_raw, traits, text_input)
    render_args['uploaded_filename'] = filename
    return render_template('text_dash.html', **render_args)


if __name__ == '__main__':
    app.run(debug=True)
