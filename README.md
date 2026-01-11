Audio Digit Recognition MLP

Hi, this is a audio-digit recognition MLP that I created from scratch and implemented in Python using CuPy, NumPy, librosa, and other libraries.

The model recognizes spoken digits from one to nine. When a user speaks a single digit, the system records the live audio, extracts features, runs them through the MLP, and outputs the predicted digit in the application.

The logic of the MLP was taken from my CS 440 notes and the internet. The data used to train the model is from googles speech command dataset on kaggle https://www.kaggle.com/datasets/yashdogra/speech-commands?resource=download. 

I chose digit recognition because it paid homage to the classic digit image recognition problem shown in CS 440 (Fall 2025) in Professor Lirong’s class, while staying within my computational and data gathering limits.

Training was done through Rutgers ilab machines using their gpus since my personal computer is throttled at the moment, thank you Rutgers ilab! Trained weights are saved in weights.npz. 

The model is around 90% accurate and I hope you try it yourself. To use test out the model yourself please clone the git repository, create a virtual environment, extract dependencies using "pip install ." and run "python -m get_aud". 

This will open up a minimal GUI made in tkinter that will allow you to record your own voice saying a digit. When you first press record, the GUI may briefly say “Not Responding”. This is expected do not click again just wait a moment and it will fix itself.

Please press record only once per attempt and wait around 0.5-1 seconds before speaking, or else the audio will not capture correctly due to loading time. 

The prediction of the number you said will be displayed on the GUI.

If you need to change the input microphone, the "1" in "sd.default.device = (1, 4)" to your desired microphone input index on the main of get_aud.py. Microphone indexes are printed to the terminal when get_aud.py is ran.

Here is me running the code!
https://youtu.be/mvJFuRyheho



