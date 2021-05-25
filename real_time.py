from tensorflow.keras.models import load_model
import pyaudio
import struct
import time
import librosa
import numpy as np
from matplotlib import cm
import pylab
import cv2
import librosa.display
import matplotlib.pyplot as plt

model = load_model("cough_detector.model")

import microphones
desc, mics, indices = microphones.list_microphones()
print(mics)
MICROPHONE_INDEX = indices[20]


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
stream.start_stream()
while True:
	data = stream.read(CHUNK)  
	#print(len(data))
	# convert data to integers, make np array
	data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

	# create np array and offset by 128
	data_np = (np.array(data_int, dtype='b')).astype('float32')
	length = data_np.shape[0] / RATE
	#print(length)
	# Compute spectrogram
	melspec = librosa.feature.melspectrogram(y=data_np, sr=44100, n_mels=128)

	log_melspec = librosa.power_to_db(melspec, ref=np.max)  
	librosa.display.specshow(log_melspec, sr=44100)
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
	preds_cough=model.predict(new_array)[0]
	(clapping, coughing, crying_baby, laughing, sneezing)= preds_cough
	if np.max(preds_cough)==coughing:
		print('cough=', coughing)
	else:
		print('not cough')

	





