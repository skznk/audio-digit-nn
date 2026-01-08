import sounddevice as sd
from tkinter import *
from tkinter import ttk
import nnaud as n


livenet = n.Neural_Net()

def record(duration=2): #send this to nnaud
    
    myrecording = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
    sd.wait()
    num_to_word(livenet.inference(n.extract_normalized_features(file_path=None, live_audio=myrecording[:, 0].astype("float32", copy=False)))[1])

def num_to_word(number):
    labels = {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine"}

    dynamic_str.set('You said: ' + labels[number])
    

if __name__ == '__main__':
    print(sd.query_devices())
    sd.default.device = (1, 4) #Change to any other index based on previous lines print (input, output)

    root = Tk()
    root.title('NN - Speech Recognition')
    root.geometry('400x300')
    root.resizable(False,False)

    dynamic_str = StringVar(value='')

    text_label = Label(root, textvariable=dynamic_str)
    text_label.pack(expand=True)

    recbutton = ttk.Button(root, text='Record', command=record)
    recbutton.pack(expand=True)

    root.mainloop()
