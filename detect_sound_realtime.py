from tensorflow.keras.models import load_model
import pyaudio
import struct
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import librosa.display

model = load_model("cough_detector.model")

import microphones
desc, mics, indices = microphones.list_microphones()
print(mics)
MICROPHONE_INDEX = indices[1]


# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    
    i = indices[k]
    #print(i)
    if (i==MICROPHONE_INDEX):
        print(i)
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

# constants
CHUNK = 55125            # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

print("# Live Prediction Using Microphone: %s" % (mic_desc))
stream.start_stream()
#time.sleep(2.0)
print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
    # binary data
    data = stream.read(CHUNK)  
    #print(len(data))
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    
    # create np array and offset by 128
    data_np = (np.array(data_int, dtype='b')).astype('float32')
    length = data_np.shape[0] / RATE
    print(length)
    # Compute spectrogram
    melspec = librosa.feature.melspectrogram(y=data_np, sr=44100, n_mels=128)
# Convert to log scale (dB) using the peak power (max) as reference
    # per suggestion from Librbosa: https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  
    librosa.display.specshow(log_melspec, sr=44100)
    #file_name='pos-0421-084-cough-m-50-1.mp3'
    plt.savefig('record.png')
    
    
    img_array = cv2.imread('record.png')  # convert to array
    new_array = cv2.resize(img_array, (224, 224))  # resize to normalize data size
    #new_array=np.array(new_array).reshape(-1, 224, 224, 3)
    #new_array = img_to_array(new_array)
    #new_array = preprocess_input(new_array)
    new_array=np.array(new_array).reshape(-1, 224, 224, 3)
    new_array = np.array(new_array, dtype="float32")
    #print(new_array.shape)
    #new_array = np.expand_dims(new_array)
    print(model.predict(new_array))
            
   