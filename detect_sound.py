#from scipy.io import wavfile
#from pydub import AudioSegment 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from tensorflow.keras.models import load_model
import cv2
#samplerate, data = wavfile.read('1-19111-A-24.wav')
#samples, sample_rate = sf.read('1-19111-A-24.wav')
model = load_model("cough_detector.model")
data,rate = librosa.core.load('pos-0421-084-cough-m-50-1.mp3', sr=44100)
            #print(rate)
#wav_file = wav_file.split_to_mono() 

#length = data.shape[0] / rate
#print(f"number of channels = {data.shape}")
# Pre-emphasis filter
melspec = librosa.feature.melspectrogram(y=data, sr=44100, n_mels=128)
# Convert to log scale (dB) using the peak power (max) as reference
    # per suggestion from Librbosa: https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
log_melspec = librosa.power_to_db(melspec, ref=np.max)  
librosa.display.specshow(log_melspec, sr=44100)
file_name='pos-0421-084-cough-m-50-1.mp3'
plt.savefig(file_name.strip('.mp3') + '.png')


img_array = cv2.imread('os-0421-084-cough-m-50-1.png')  # convert to array
new_array = cv2.resize(img_array, (224, 224))  # resize to normalize data size
#new_array=np.array(new_array).reshape(-1, 224, 224, 3)
#new_array = img_to_array(new_array)
#new_array = preprocess_input(new_array)
new_array=np.array(new_array).reshape(-1, 224, 224, 3)
new_array = np.array(new_array, dtype="float32")
#print(new_array.shape)
#new_array = np.expand_dims(new_array)
print(model.predict(new_array))
#
# =============================================================================
