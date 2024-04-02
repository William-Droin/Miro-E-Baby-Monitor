import numpy as np
import requests
import librosa
from tensorflow import keras
import time
import io
import os

# Hide warnings related to TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# If the model file exists, load it
model = keras.models.load_model('/Users/williamdroin/Desktop/Miro-E-Baby-Monitor/model.keras')

# Define parameters
target_sr = 20000  # Sample rate to match RATE in your audio streaming server
n_mfcc = 40  # Number of MFCC coefficients
stream_url = 'http://localhost:5001/audio'  # URL of your audio stream

def preprocess_audio_mfcc(audio_data, sr, n_mfcc):
    # Assuming audio_data is a 1D NumPy array
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    return mfccs

while True:
    # Make a request to the audio stream
    response = requests.get(stream_url, stream=True)

    # Download a 10-second snippet of the audio stream
    snippet_length = target_sr * 10 * 2  # 10 seconds of 16-bit samples
    audio_snippet = response.raw.read(snippet_length)
    response.close()

    # Convert snippet to a NumPy array and normalize
    audio_data = np.frombuffer(audio_snippet, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max

    # Ensure the snippet is the correct length
    if len(audio_data) == target_sr * 10:
        # Preprocess and predict using the model
        mfccs = preprocess_audio_mfcc(audio_data, target_sr, n_mfcc)
        prediction = model.predict(mfccs)
        class_label = 1 if prediction > 0.5 else 0  # Adjust the threshold as needed
        print("Predicted Class Label:", class_label)
        print("Prediction Score:", prediction)
    else:
        print("Audio snippet was not the correct length. Expected", target_sr * 10, "got", len(audio_data))

    # Wait 20 seconds before processing the next snippet
    time.sleep(20)
