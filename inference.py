import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
X_train, Y_train = read_csv('data/train_emoji.csv')

maxLen = len(max(X_train, key=len).split())

class Emojifier(nn.Module):
    def __init__(self,input_dim, emb_matrix,layer_dim, hidden_dim=128,output_dim=5,drp=0.5):
        super().__init__()
        self.drp=drp
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.layer_dim=layer_dim
        self.lstm= nn.LSTM(emb_dim,hidden_dim,layer_dim,batch_first=True,dropout=self.drp)
        self.fc= nn.Linear(hidden_dim,output_dim)
        self.embedding = nn.Embedding(vocab_len, emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix,dtype=torch.float32))
        self.embedding.weight.requires_grad = False #not trainable
        self.dropout = nn.Dropout(drp)

    def forward(self,x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim,device='cuda').requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim,device='cuda').requires_grad_()
        x   =   self.embedding(x)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

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
	for i in range(2):  # give cuda more time to load (brute force way to fix CUDA 9 and RTX conflicts)
		try:
			model = Emojifier(emb_matrix.shape[1], emb_matrix, 2).to(device)
		except RuntimeError as e:
			pass
	out=model(sentence)
	result = label_to_emoji(int(torch.argmax(out)))
	print(result)
	
if __name__ == '__main__':

	model = load_checkpoint('lstm_checkpoint.pth')
	print('model loaded!!')
	predict()
