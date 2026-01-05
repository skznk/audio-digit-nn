from pydub import AudioSegment
from pathlib import Path
import shutil
import os
import random
import sys

def conToWav(path_to_folder): # Converts unprocessed audio files in path folder to wav audio files then places them in the wav folder

    #-------------------------------------------------------- This code is for saving the file index in \all_audio_files directory so we don't convert the same file multiple times
    FILE_INDEX = 0 
    with open('ind.txt',mode='r') as f:
        ind = f.read()
        FILE_INDEX = int(ind)
    #---------------------------------------------------------------

    ap = Path(path_to_folder)
    listofFiles = [(x.name, x.suffix, x) for x in ap.iterdir()] # List of all the files in \all_audio_files
    listofFiles.sort(key=lambda x:x[0])

    for x,y in enumerate(listofFiles[FILE_INDEX:]):     #Converts audio files to wav
        try:
            if y[1] == '.mp4':
                FILE_INDEX = max(FILE_INDEX,x+1)
                input = AudioSegment.from_file(str(y[2]), format = 'mp4')
                input.export("C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\"+y[0], format='wav')
            elif y[1] == '.mp3':
                FILE_INDEX = max(FILE_INDEX,x+1)
                input = AudioSegment.from_file(str(y[2]), format = 'mp3')
                input.export("C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\"+y[0], format='wav')
            elif y[1] == '.mpeg-4':
                FILE_INDEX = max(FILE_INDEX,x+1)
                input = AudioSegment.from_file(str(y[2]), format = 'mpeg-4')
                input.export("C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\"+y[0], format='wav')
            elif y[1] == '.wav':
                FILE_INDEX = max(FILE_INDEX,x+1)
                shutil.copy(str(y[[2]]),"C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\"+y[0])
            else: # return
                print("Unidentified file format")
                raise Exception
        except Exception as e:
            print(e)
            with open('ind.txt',mode='w') as f:
                f.write(str(FILE_INDEX))

    with open('ind.txt',mode='w') as f:
        f.write(str(FILE_INDEX)) #Writes index to ind.txt

def splitInto(path_to_folder, train_prob, val_prob, test_prob): #places audio wav files into train, val, or test
    #path to digits folder in wav
    #FILE_INDEX = 0 
    #with open('ind.txt',mode='r') as f:
    #    ind = f.read()
    #    FILE_INDEX = int(ind)
    #---------------------------------------------------------------

    all_digit_paths = [Path(path_to_folder+'\\zero'),Path(path_to_folder+'\\one'),Path(path_to_folder+'\\two'),Path(path_to_folder+'\\three'),Path(path_to_folder+'\\four'),Path(path_to_folder+'\\five'),Path(path_to_folder+'\\six'),Path(path_to_folder+'\\seven'),Path(path_to_folder+'\\eight'),Path(path_to_folder+'\\nine')]
    listofFiles = [[(x.name, x) for x in y.iterdir()] for y in all_digit_paths] # List of all the files in each numbers folder
    
    for i in range(len(all_digit_paths)):
        listofFiles[i].sort(key=lambda x:x[0])
    
    num = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine']
    trainprob = 0.7
    valprob = 0.15
    testprob = 0.15

    for i in range(len(listofFiles)):
        for x,y in enumerate(listofFiles[i]):     #places audio wav files into train, val, or test
            try:
                prob = random.uniform(0.0,1.0)
                print(prob)
                print(x)
                print(y)
                print(i)
                if prob < trainprob:
                    shutil.copy(str(y[1]),"C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\train\\"+num[i]+'\\'+y[0])
                elif trainprob <= prob < trainprob+valprob:
                    shutil.copy(str(y[1]),"C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\val\\"+num[i]+'\\'+y[0])
                else:
                    shutil.copy(str(y[1]),"C:\\Users\\sarmi\\daudiorec\\wav_audio_files\\test\\"+num[i]+'\\'+y[0])

            except Exception as e:
                print(e)
                sys.exit()
                

   
        

if __name__ == '__main__':
    splitInto(r"C:\Users\sarmi\daudiorec\wav_audio_files\digits", 0,0,0)
    # conToWav(r'C:\Users\sarmi\daudiorec\all_audio_files')

