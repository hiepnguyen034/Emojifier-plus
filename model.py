import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from model import Emojifier

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())

Y_train = torch.LongTensor(Y_train)

def train_model(batch_size=32, learning_rate=0.1, epoches=700):
	torch.cuda.current_device()
	for i in range(2):  # give cuda more time to load (brute force way to fix CUDA 9 and RTX conflicts)
	    try:
	        model = Emojifier(emb_matrix.shape[1], emb_matrix, 2).to(device)
	    except RuntimeError as e:
	        print('this error will go away after the first iteration:', e)
	X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
	X_train_indices = X_train_indices.astype(int)
	X_train_indices = torch.LongTensor(X_train_indices)
	n, c = X_train_indices.shape
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	for epoch in range(epoches):
		for i in range((n - 1) // batch_size + 1):
			start_i = i * batch_size
			end_i = start_i + batch_size
			xb = X_train_indices[start_i:end_i].to(device)
			yb = Y_train[start_i:end_i].to(device)
			pred = model(xb)
			loss = criterion(pred, yb).to(device)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		if epoch % 50 == 0:
		    print('cost after iteration',epoch,':',criterion(model(xb), yb).item())
	X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
	X_test_indices=X_test_indices.astype(int)
	X_test_indices=torch.LongTensor(X_test_indices)
	pred = model(X_test_indices.to(device)).detach().cpu().numpy()
	y_pred=[]
	for i in range(len(X_test)):
	    num = np.argmax(pred[i])
	    y_pred.append(num)
	print('current accuracy: ',accuracy_score(y_pred,Y_test))
	checkpoint = {'model': Emojifier(emb_matrix.shape[1], emb_matrix, 2).to(device),
	              'state_dict': model.state_dict(),
	              'optimizer': optimizer.state_dict(),
	              'epoches': epoches}
	torch.save(checkpoint, 'lstm_checkpoint.pth')


if __name__ == '__main__':
    train_model()
