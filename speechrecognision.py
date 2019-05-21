import speech_recognition as sr
# Search for available microphones
# for index, name in enumerate(sr.Microphone.list_microphone_names()):
#     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

recording = sr.Recognizer()

with sr.Microphone(device_index=1) as source:
    recording.adjust_for_ambient_noise(source)
    print("please say something:")
    audio = recording.listen(source)
try:
    print("You said: \n" + recording.recognize_google(audio))
except Exception as e:
    print(e)
