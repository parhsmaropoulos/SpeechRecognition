import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

# Read file and return 2 values
frequency_sampling, audio_signal = wavfile.read("D:/Users/Parhs/Σχολη/ΦωνηΚαιΗχος/audios/harvard.wav")

# Display the parameters below
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
# print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')

# Normalize the signal
# audio_signal = audio_signal / np.power(2, 15)

# # # # VISUALIZE AUDIO SIGNAL
# # # Extracting 100 first values
# #
# # audio_signal = audio_signal[:100]
# # time_axis = 1000*np.arange(0, len(audio_signal), 1) / float(frequency_sampling)
# #
# # # Visual signal
# #
# # plt.plot(time_axis, audio_signal, color='blue')
# # plt.xlabel('time(millisecond)')
# # plt.ylabel('Amplitude')
# # plt.title('Input audio signal')
# # plt.show()

# # #  CHARACTERIZING AUDIO SIGNAL
# # Length and half-length of the signal
#
# length_signal = len(audio_signal)
# half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)
#
# # Apply maths to transform it into frequency domain(Fourier Transform)
#
# signal_frequency = np.fft.fft(audio_signal)
#
# # Normalize and squaring
#
# signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
# signal_frequency **=2
#
# # Extract length/half_length
#
# len_fts = len(signal_frequency)
#
# # Adjust fourier transformed signal
#
# if length_signal % 2:
#     signal_frequency[1:len_fts] *=2
# else:
#     signal_frequency[1:len_fts+1] *=2
#
# # Extract power in dB
#
# signal_power = 10* np.log10(signal_frequency)
#
# # Adjust frequency in kHZ for X-axis
#
# x_axis = np.arange(0, half_length , 1) * (frequency_sampling/ length_signal) / 1000.0
#
# # Visualize characterization
#
# plt.figure()
# plt.plot(x_axis, signal_power, color='black')
# plt.xlabel('time(millisecond)')
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Signal power (dB)')
# plt.show()

# FEAUTURE EXCTRACTION

audio_signal = audio_signal[:15000]

#Extract Mel Frequency Cepstral Coefficient(MFCC) feature

features_mfcc = mfcc(audio_signal,frequency_sampling)

# Print the paramenters

print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

# Plot and visualize mfcc features

features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

# Extract filter bank feature

filterbank_features = logfbank(audio_signal, frequency_sampling)

# Print filterbank param

print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

# Plot and Visualize filterbank features

filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()




