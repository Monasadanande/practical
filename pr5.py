# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 23:07:51 2025

@author: monas
"""

"""pr5"""

import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
%matplotlib inline

sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)   # remove special chars
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()  # remove 1-letter words
sentences = sentences.lower()  # lowercase all text

words = sentences.split()
vocab = set(words)
vocab_size = len(vocab)
embed_dim = 10
context_size = 2

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

data = []
for i in range(2, len(words)-2):
    context = [words[i-2], words[i-1], words[i+1], words[i+2]]
    target = words[i]
    data.append((context, target))

embeddings = np.random.random_sample((vocab_size, embed_dim))

def linear(m, theta):
    return m.dot(theta)

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum() / len(out)

def log_softmax_crossentropy_with_logits(logits, target):
    out = np.zeros_like(logits)
    out[np.arange(len(logits)), target] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (-out + softmax) / logits.shape[0]

def forward(context_idxs, theta):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)
    return m, n, o

def backward(preds, theta, target_idxs):
    m, n, o = preds
    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)
    return dw

def optimize(theta, grad, lr=0.03):
    theta -= grad * lr
    return theta

theta = np.random.uniform(-1, 1, (2*context_size*embed_dim, vocab_size))
epoch_losses = {}

for epoch in range(80):
    losses = []
    for context, target in data:
        context_idxs = np.array([word_to_ix[w] for w in context])
        preds = forward(context_idxs, theta)
        target_idxs = np.array([word_to_ix[target]])
        loss = NLLLoss(preds[-1], target_idxs)
        losses.append(loss)
        grad = backward(preds, theta, target_idxs)
        theta = optimize(theta, grad, lr=0.03)
    epoch_losses[epoch] = losses

plt.figure()
plt.title("Epoch vs Loss")
plt.plot(range(80), [epoch_losses[i][0] for i in range(80)])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

def predict(words):
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    return ix_to_word[np.argmax(preds[-1])]

def accuracy():
    wrong = 0
    for context, target in data:
        if predict(context) != target:
            wrong += 1
    return 1 - (wrong / len(data))

accuracy()  # → 1.0

