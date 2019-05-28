#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(len(vocab.char2id),embed_size)
        self.embed_size = embed_size
        self.dropout = nn.Dropout(0.3)
        self.highway = Highway(embed_size)
        self.cnn = CNN(embed_size)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1j
        x_emb = self.embeddings(input)
        x_convout = self.cnn(x_emb.reshape(-1,x_emb.size()[3],x_emb.size()[2]))
        #print(x_convout.size())
        x_highway = self.highway(torch.squeeze(x_convout))
        x_highway = x_highway.reshape(x_emb.size()[0],x_emb.size()[1],-1)
        return self.dropout(x_highway)
        ### END YOUR CODE

