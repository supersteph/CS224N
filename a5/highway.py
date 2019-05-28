#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_conv):
        x_proj = self.relu(self.projection(x_conv))
        x_gate = self.sigmoid(self.gate(x_conv))
        return x_proj*x_gate+(torch.ones(self.embed_size)-x_gate)*x_conv
### END YOUR CODE 

