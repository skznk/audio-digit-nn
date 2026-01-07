import librosa
from librosa import feature
import numpy as np
import cupy as cp
from pathlib import Path
import random


class Neural_Net(): # 5 layer Neural Network  I'm initializing to this 60->128->64->32->10 

    def __init__(self):
        self.layer0_neurons = 60 #amount of input neurons for each layer
        self.layer1_neurons = 128
        self.layer2_neurons = 64
        self.layer3_neurons = 32
        self.outputs = 10

        self.l0_1_weights = self.initializeWeights(60, 128)
        self.l1_2_weights = self.initializeWeights(128, 64)
        self.l2_3_weights = self.initializeWeights(64, 32)
        self.l3_4_weights = self.initializeWeights(32, 10)
        self.bias = [ cp.zeros((128, 1), dtype=cp.float32), cp.zeros((64, 1), dtype=cp.float32), cp.zeros((32, 1), dtype=cp.float32), cp.zeros((10, 1), dtype=cp.float32)]


        self.layer0_inp = 0 #inputs during inference for a label
        self.layer1_inp = 0
        self.layer2_inp = 0
        self.layer3_inp = 0
        self.check_saved_weights()


    def initializeWeights(self,prev_layer_outputs, layer_inputs): # Uses He initialization for neural networks based on layers being  60->128->64->10
        #std0_in = np.sqrt(2/layer0_inputs) #60 inputs
        #std1_in = np.sqrt(2/layer1_inputs) #128 inputs
        #std2_in = np.sqrt(2/layer2_inputs) #64 inputs
        #std3_in = np.sqrt(2/layer3_inputs) #32 inputs
        
        std_in = cp.sqrt(2/prev_layer_outputs)
            
        return cp.random.normal(loc=0.0, scale=std_in, size=(layer_inputs, prev_layer_outputs))
    
    def check_saved_weights(self):
        
        if Path("weights.npz").exists():
            self.loadWeights()
        else:
            print('no saved weights')

    def inference(self, initial_features, batch_size): #arr should be [weight0_arr, weight1_arr, weight2_arr, weight3_arr]
        
        def softmax(final_layer):

            fin_max = cp.max(final_layer, axis=0, keepdims=True)
            e_powered = cp.exp(final_layer-fin_max)
            e_sum = cp.sum(e_powered,axis=0,keepdims=True)
            
            return e_powered/e_sum
        
       
        self.layer0_inp = initial_features

        l0_1result = cp.matmul(self.l0_1_weights, initial_features) + self.bias[0] #output is 128 dimensional
        l0_1ReLU = cp.maximum(0, l0_1result) #ReLU on layer 0
        
        self.layer1_inp = l0_1ReLU
        l1_2result = cp.matmul(self.l1_2_weights, l0_1ReLU) + self.bias[1] #output is 64 dimensional
        l1_2ReLU = cp.maximum(0, l1_2result) #ReLU on layer 1
        
        self.layer2_inp = l1_2ReLU
        l2_3result = cp.matmul(self.l2_3_weights, l1_2ReLU) + self.bias[2]  #output is 32 dimensional
        l2_3ReLU = cp.maximum(0, l2_3result) #ReLU on layer 2
        
        self.layer3_inp = l2_3ReLU
        l3_4result = cp.matmul(self.l3_4_weights, l2_3ReLU) + self.bias[3] #output is 10 dimensional

        soft = softmax(l3_4result)

        if soft.shape[1] == 1:
            return (soft, self.label_guessed(softmax_in=soft))
        elif soft.shape[1] > batch_size:
            return (soft, self.label_guessed(softmax_in=soft,val=1))


        return (soft, 0) #10 dimensional vector 0-9

    def label_guessed(self, softmax_in, val=0):
        
        if val==1:
            return cp.argmax(softmax_in, axis=0).astype(cp.int32).get().tolist()
        
        return int(cp.argmax(softmax_in, axis=0).get()[0])
    
    def backprop(self, fin_output, correct_label_ind, learning_rate):
        
        #output gradient
        batchsize = fin_output[0].shape[1]
        
        #gvec = cp.zeros((self.outputs,1), dtype=cp.float32)
        gstack = cp.zeros_like(fin_output[0])
        gstack[correct_label_ind, cp.arange(batchsize)] = 1

        error_signal_4 = fin_output[0] - gstack # error signal pL/pZ
        grad_weights_4 = cp.matmul(error_signal_4, cp.transpose(self.layer3_inp)) #pdL/pDW
        grad_weights_4/=batchsize
        
        self.bias[3] = self.bias[3] - learning_rate* cp.mean(error_signal_4,axis=1,keepdims=True)

        #prev layer grad
        error_signal_3 = cp.matmul(cp.transpose(self.l3_4_weights),error_signal_4) * Neural_Net.activation_derivative(self.layer3_inp)
        grad_weights_3 = cp.matmul(error_signal_3,cp.transpose(self.layer2_inp))
        grad_weights_3/=batchsize

        self.bias[2] = self.bias[2] - learning_rate* cp.mean(error_signal_3,axis=1,keepdims=True)

        #prev layer grad
        error_signal_2 = cp.matmul(cp.transpose(self.l2_3_weights),error_signal_3) * Neural_Net.activation_derivative(self.layer2_inp)
        grad_weights_2 = cp.matmul(error_signal_2, cp.transpose(self.layer1_inp))
        grad_weights_2/=batchsize

        self.bias[1] = self.bias[1] - learning_rate*cp.mean(error_signal_2,axis=1,keepdims=True)

        #prev layer grad
        error_signal_1 = cp.matmul(cp.transpose(self.l1_2_weights),error_signal_2) * Neural_Net.activation_derivative(self.layer1_inp)
        grad_weights_1 = cp.matmul(error_signal_1, cp.transpose(self.layer0_inp))
        grad_weights_1/=batchsize

        self.bias[0] = self.bias[0] - learning_rate*cp.mean(error_signal_1,axis=1,keepdims=True)
        
        self.l3_4_weights = self.l3_4_weights - learning_rate*grad_weights_4
        self.l2_3_weights = self.l2_3_weights - learning_rate*grad_weights_3
        self.l1_2_weights = self.l1_2_weights - learning_rate*grad_weights_2
        self.l0_1_weights = self.l0_1_weights - learning_rate*grad_weights_1

       

    def activation_derivative(value):
        
        return (value>0).astype(cp.float32)

    def train(self, batch=100, epoch=300, learning_rate=0.001):
        all_batch_matrices = create_batch_matrices(batch)
        for i in range(epoch):
            random.shuffle(all_batch_matrices)
            print('Training '+ str((i/epoch)*100)+ '% Complete: Epoch '+ str(i))
            for j in range(len(all_batch_matrices)):
                self.backprop(self.inference(initial_features=all_batch_matrices[j][0],batch_size=batch), all_batch_matrices[j][1], learning_rate)
            print('Validation is '+str(self.runval(batch)*100) + '% Correct')

        self.saveWeights()

    def runval(self, batch):
        files = get_all_paths(path_to_dir=r'/common/home/sn887/audio-digit-nn/wav_audio_files/val')
        labels = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}

        tlist = [extract_normalized_features(file_path=f) for f in files]
        labels = [labels[f.parent.name] for f in files]
        
        x = cp.concatenate(tlist,axis=1)

        guessedlabs = self.inference(initial_features=x,batch_size=batch)[1]

        return sum(x == y for x,y in zip(labels,guessedlabs))/len(labels)
    

    def saveWeights(self): #saves weights and biases from training
        
        cp.savez('savedfeatures/weights.npz', W1=self.l0_1_weights, W2=self.l1_2_weights, W3=self.l2_3_weights, W4=self.l3_4_weights, b1=self.bias[0], b2=self.bias[1], b3=self.bias[2], b4=self.bias[3])

    def loadWeights(self):
        data = cp.load('weights.npz')

        self.l0_1_weights = data["W1"]
        self.l1_2_weights = data["W2"]
        self.l2_3_weights = data["W3"]
        self.l3_4_weights = data["W4"]

        self.bias[0] = data["b1"]
        self.bias[1] = data["b2"]
        self.bias[2] = data["b3"]
        self.bias[3] = data["b4"]


def collect_and_save_features(filepath='/common/home/sn887/audio-digit-nn/wav_audio_files/train'): #speeds up time of training by just storing the features instead of computing them every single epoch
    
    labels = {"zero":'savedfeatures/zero/',"one":'savedfeatures/one/',"two":'savedfeatures/two/',"three":'savedfeatures/three/',"four":'savedfeatures/four/',"five":'savedfeatures/five/',"six":'savedfeatures/six/',"seven":'savedfeatures/seven/',"eight":'savedfeatures/eight/',"nine":'savedfeatures/nine/'}

    stuff = get_all_paths()
    for i in range(len(stuff)):
        cp.savez(labels[stuff[i].parent.name]+stuff[i].stem+'.npz', feature=extract_normalized_features(stuff[i]))

    

def extract_features(file=None, raw=None):
            try:
                if file is not None:
                    y,sr = librosa.load(file) # sampling rate default to 22050 Hz
                    fts = feature.mfcc(y=y, sr=sr, hop_length = 176, win_length = 529) # We are calculating the feature for each frame. (frame length - hop length)/frame length is the overlap percentage
                    deltaf1 = feature.delta(data=fts, order=1) #20 features
                    deltaf2 = feature.delta(data=fts, order=2) #20 features
                    final = (fts.mean(axis=1), deltaf1.mean(axis=1), deltaf2.mean(axis=1)) #60 features
                    tim = np.array([*final[0],  *final[1] , *final[2]]).reshape(-1, 1)
                    return cp.asarray(tim)
                else:
                    sr = 22050 
                    fts = feature.mfcc(y=raw, sr=sr, hop_length = 176, win_length = 529) # We are calculating the feature for each frame. (frame length - hop length)/frame length is the overlap percentage
                    deltaf1 = feature.delta(data=fts, order=1) #20 features
                    deltaf2 = feature.delta(data=fts, order=2) #20 features
                    final = (fts.mean(axis=1), deltaf1.mean(axis=1), deltaf2.mean(axis=1)) #60 features
                    tim = np.array([*final[0],  *final[1] , *final[2]]).reshape(-1, 1)
                    
                    return cp.asarray(tim)


            except Exception as e:
                print(e)
    
def extract_normalized_features(file_path=None, live_audio=None):

        stats = cp.load("stats.npz")
        mean, stdev = stats["mean"], stats["stdev"]
        if file_path is not None: #for training
            x = extract_features(file=file_path).astype(cp.float32)     
            x = (x - mean) / stdev             
            
            return x
        else: #for live audio
            x = extract_features(raw=live_audio).astype(cp.float32)     
            x = (x - mean) / stdev             
            
            return x


def normalize_features():

        file_names = get_all_paths()
        cp_matrix = cp.concatenate([extract_features(file=x) for x in file_names], axis=1) #(numfiles,60) matrix
        
        mean =  cp.mean(cp_matrix, axis=1, keepdims=True)       
        stdev = cp.std(cp_matrix, axis=1, keepdims=True) 
        
        cp.savez("stats.npz", mean=mean, stdev=stdev)

        return mean, stdev

def get_stats():
    stats = cp.load("stats.npz")
    mean, stdev = stats["mean"], stats["stdev"]
    print('mean')
    print(mean)
    print('standard deviation')
    print(stdev)
    
def get_all_paths(path_to_dir=r'/common/home/sn887/audio-digit-nn/wav_audio_files/train'): #path to all wav audio files in train dir
        
        ap = Path(path_to_dir)
        listof_folders = [Path(x) for x in ap.iterdir()]
        list_of_all_files = [x for f in listof_folders for x in f.iterdir()]

        return list_of_all_files

def create_batch_matrices(batchsize): #create batch matrices and labels
    labels = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}

    stuff = get_all_paths()
    batches = []
    for j in range(0, len(stuff), batchsize):
        files = stuff[j:j+batchsize]
        feats = [cp.load(str(Path('/common/home/sn887/audio-digit-nn/savedfeatures/'+f.parent.name)/Path(f.stem+'.npz')))['feature'] for f in files]   
        labs = [labels[f.parent.name] for f in files]                  

        x = cp.concatenate(feats, axis=1)                                  
        y = cp.asarray(labs, dtype=cp.int32).reshape(-1)                            
        batches.append((x, y))

    return batches
        

    

if __name__ == "__main__":
    tim_net = Neural_Net()
    tim_net.train(batch=100,epoch=600)
    
    

    