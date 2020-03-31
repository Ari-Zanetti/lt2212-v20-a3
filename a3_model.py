import os
import sys
import argparse
from itertools import combinations
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.utils import shuffle
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as pyplot

layers = {1: nn.ELU, 2: nn.LogSigmoid}

class PerceptronModel(nn.Module):
    
    def __init__(self, inputsize, hiddenlayer_type=None, hiddenlayer_size=0):
        super(PerceptronModel, self).__init__()       
        self.hidden = None
        self.linear2 = None    
        if hiddenlayer_size:
            self.linear0 = nn.Linear(inputsize, hiddenlayer_size)
            if hiddenlayer_type:
                self.hidden = layers[hiddenlayer_type]()
            self.linear2 = nn.Linear(hiddenlayer_size, 1)
        else:
            self.linear0 = nn.Linear(inputsize, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear0(x)
        if self.hidden:
            x = self.hidden(x)
        if self.linear2:
            x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    
    def predict(self, row):
        sample1 = [eval(x) for x in row[0]]
        sample2 = [eval(x) for x in row[1]]
        sample1.extend(sample2)
        instance = torch.Tensor(sample1)
        return self(instance)


def __sample_data(df):
    samples = list(combinations(df.values, 2))
    df = pd.DataFrame(samples)
    df = df.stack()
    y = []
    df2 = df.unstack()
    y = df2.apply(lambda row: row[0][-1]==row[1][-1], axis=1)
    df = df.apply(lambda row: row[1:-2])
    df = df.unstack()
    df.insert(0,"class",y)
    
    positive_samples = df[df["class"] == True]
    negative_samples = df[df["class"] == False][:positive_samples.shape[0]]
    samples = pd.concat([positive_samples,negative_samples])
    return samples


def __train_and_test_model(model, epochs, X, X_test, thresholds):
    print("Perceptron: {}".format(model))
    model.zero_grad()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 0.001)
    for i in range(epochs):
        if epochs > 1 :
            print("Running epoch {}...".format(i+1))
        X = shuffle(X)
        for i, row in X.iterrows():
            label_predict = model.predict(row)
            label = torch.FloatTensor([1 if row["class"] == True else 0])
        
            loss = criterion(label_predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print("Training complete.")
    results = {}
    for t in thresholds:
        y=[]
        y_pred=[]
        for i, row in X_test.iterrows():
            label_predict = model.predict(row)
            label = torch.FloatTensor([1 if row["class"] == True else 0])
            pred_val = 0 if label_predict < t else 1
            y_pred.append(pred_val)
            y.append(label)
    
        precision = metrics.precision_score(y, y_pred, average='binary')
        recall = metrics.recall_score(y, y_pred, average='binary')
        results[t] = (precision, recall)
        
        print("Accuracy with treshold {}:".format(t), metrics.accuracy_score(y, y_pred))
        print("Precision with treshold {}:".format(t), precision)
        print("Recall with treshold {}:".format(t), recall)
        print("F-measure with treshold {}:".format(t), metrics.f1_score(y, y_pred,average='binary'))
    
    return results
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("-e", "--epochs", nargs='?', help="Number of epochs.")
    parser.add_argument("-nl", "--nonlinearity", nargs='?', choices={"1", "2"}, help="The chosen non-linearity, 1 or 2.")
    parser.add_argument("-size", "--size", nargs='?', help="Size of the hidden layer.")
    parser.add_argument("-plot", "--plot", nargs='?', help="Name of the output file for the precision-recall curve plot for your model.")
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    
    corpus = []
    with open(args.featurefile) as f:
        for line in f.readlines():
            row = line.split(",")
            row = row[:-1]
            corpus.append(row)

    df = np.array(corpus)
    df_train = df[df[:,-2]=='0.0']
    df_test = df[df[:,-2]!='0.0']
    X = __sample_data(pd.DataFrame(df_train))
    print("Generated {} samples for training.".format(X.shape[0]))
    X_test = __sample_data(pd.DataFrame(df_test))
    print("Generated {} samples for test.".format(X_test.shape[0]))
    
    print("Train sample: {}".format(X.iloc[0]))
    
    inputsize = len(X[0][0]) * 2
    epochs = 1
    nonlinearlayer = None
    hiddenlayersize = None
    if args.epochs:
        epochs = eval(args.epochs)
        
    if args.size:
        if args.nonlinearity: 
            # Add non-linear hidden layer
            nonlinearlayer = eval(args.nonlinearity)
        # Add linear hidden layer
        hiddenlayersize = eval(args.size)
        
    if not args.plot:
        print("Creating perceptron model with input", inputsize, "non linear",nonlinearlayer, "hidden size", hiddenlayersize) 
        model = PerceptronModel(inputsize, nonlinearlayer, hiddenlayersize)
        __train_and_test_model(model, epochs, X, X_test, [0.5])
    else:    
        thresholds = [0.0, 0.25, 0.5, 0.625, 1.0]
        hidden_layer_sizes = [inputsize // 2, inputsize * 2 // 3, inputsize]
        
        for s in hidden_layer_sizes:
            model = PerceptronModel(inputsize, nonlinearlayer, s)
            results = __train_and_test_model(model, epochs, X, X_test, thresholds)
            precision = [x[0] for i, x in results.items()]
            recall = [x[1] for i, x in results.items()]
            pyplot.plot(recall, precision, marker='.', label='Logistic')
        
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.savefig(args.plot)
                
                
        
    
    
