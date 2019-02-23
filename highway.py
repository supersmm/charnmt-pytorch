#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """Highway Model"""
    def __init__(self, embed_size):
        """Initiate Highway Model
        In the constructor we instantiate two nn.Linear modules and assign them as member variables.

        e_word: word embedding length
        x_conv_out (Tensor): (batch_size, e_word)
        """
        super(Highway, self).__init__()

        self.embed_size = embed_size

        self.x_proj_linear = nn.Linear(embed_size, embed_size, bias=True) # x_proj_input = W_proj * x_conv_out + b_proj
        self.x_gate_linear = nn.Linear(embed_size, embed_size, bias=True) # x_gate_input = W_gate * x_conv_out + b_gate


    def forward(self, x_conv_out) -> torch.Tensor:
        """
        Applies x_highway = x_gate ⨀ x_proj + (1 - x_gate) ⨀ x_conv_out

        @param x_conv_out (Tensor): shape (batch_size*sentence_length, embed_size)

        @returns x_highway (Tensor): shape (batch_size*sentence_length, embed_size)
        """
        x_proj_input = self.x_proj_linear(x_conv_out)
        x_gate_input = self.x_gate_linear(x_conv_out)
        x_proj = F.relu(x_proj_input) # x_proj = ReLu(x_proj_input)
        x_gate = torch.sigmoid(x_gate_input) # x_gate = sigmoid(x_gate_input)
        #ones = torch.ones(x_conv_out.size()[0], self.e_word)
        x_highway = x_gate*x_proj + (1-x_gate)*x_conv_out # torch.bmm(x_gate, x_proj) + torch.bmm(1-x_gate, x_conv_out)

        return x_highway#, x_proj, x_gate

"""
def question_1h_sanity_check():
    print ("-"*80)
    print("Running Sanity Check for Question 1h: class Highway()")
    print ("-"*80)

    print('Running test on intermediate and final shapes and types')
    _e_word = 300
    _batch_size = 16
    x_highway_expected_size = x_proj_expected_size = x_gate_expected_size = [_batch_size, _e_word]
    _x_conv_out = torch.rand(_batch_size, _e_word)
    _x_highway, _x_proj, _x_gate = Highway(embed_size = _e_word).forward(_x_conv_out)
    assert(list(_x_gate.size()) == x_gate_expected_size), "x_gate shape is incorrect:\n it should be {} but is: {}".format(x_gate_expected_size, list(_x_gate.size()))
    assert(list(_x_proj.size()) == x_proj_expected_size), "x_proj shape is incorrect:\n it should be {} but is: {}".format(x_proj_expected_size, list(_x_proj.size()))
    assert(list(_x_highway.size()) == x_highway_expected_size), "x_highway shape is incorrect:\n it should be {} but is: {}".format(x_highway_expected_size, list(_x_highway.size()))

    print('Running test on intermediate and final values')
    _x_conv_out = torch.Tensor([[0, 1], [-1, 0], [-1, 1]])
    _x_highway, _x_proj, _x_gate = Highway(embed_size = 2).forward(_x_conv_out)
    print(_x_highway)
    print(_x_proj)
    print(_x_gate)

    print("All Sanity Checks Passed for Question 1h: class Highway()!")
    print ("-"*80)


if __name__ == '__main__':
    question_1h_sanity_check()"""
### END YOUR CODE 