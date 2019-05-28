#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i

class CNN(nn.Module):

    """
    CNN module
    """
    def __init__(self, num_filters):
        super(CNN, self).__init__()
        self.num_filters = num_filters
        self.m_word = 21
        self.filter_size = 5
        self.conv = nn.Conv1d(self.num_filters,self.num_filters,self.filter_size)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool1d(self.m_word-self.filter_size+1)
    def forward(self, x_reshaped):
        x_conv = self.conv(x_reshaped)
        return self.max(self.relu(x_conv))

### END YOUR CODE

