# ffmpeg_path.py
import os
os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'
from pydub import AudioSegment
AudioSegment.converter = r'C:\ffmpeg\bin\ffmpeg.exe'
AudioSegment.ffmpeg = r'C:\ffmpeg\bin\ffmpeg.exe'
AudioSegment.ffprobe = r'C:\ffmpeg\bin\ffprobe.exe'
