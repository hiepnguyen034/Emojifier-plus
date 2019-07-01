import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from model import Emojifier

import warnings

warnings.filterwarnings("ignore")
X_train, Y_train = read_csv('data/train_emoji.csv')

maxLen = len(max(X_train, key=len).split())


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model

def predict():
	sentence=input('Sentence to convert: ')
	sentence=np.array([sentence])
	sentence=sentences_to_indices(sentence,word_to_index,maxLen)
	sentence=sentence.astype(int)
	sentence=torch.LongTensor(sentence).to(device)
	model = load_checkpoint('lstm_checkpoint.pth')
	print('model loaded!!')
	for i in range(2):  # give cuda more time to load (brute force way to fix CUDA 9 and RTX conflicts)
		try:
			out=model(sentence)
		except RuntimeError as e:
			pass
	result = label_to_emoji(int(torch.argmax(out)))
	print('emoji: ',result)
	
if __name__ == '__main__':
	predict()

