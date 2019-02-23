#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """CNN Model
    """

    def __init__(self, embed_size, k=5, e_char=50):

        super(CNN, self).__init__()
        
        self.e_char = e_char
        self.f = embed_size # e_word = f
        self.kernel_size = k # k
        
        self.conv = nn.Conv1d(in_channels=self.e_char, out_channels=embed_size, kernel_size=k, bias=True)


    def forward(self, x_reshaped) -> torch.Tensor:
        """
        @param x_reshaped: shape (sentence_length*batch_size, e_char, max_word_length)

        @returns x_conv_out (Tensor): shape (batch_size, sentence_length, f=e_word)

        """
        # (batch_size*sentence_length, f=e_word, max_word_length-k+1)
        x_conv = self.conv(x_reshaped) 
        x_conv_relu = F.relu(x_conv)
        x_conv_out = torch.max(x_conv_relu, dim=-1)[0] # max pool (batch_size*sentence_length, embed_size)

        return x_conv_out#, x_conv

"""
def question_1i_sanity_check():
    print ("-"*80)
    print("Running Sanity Check for Question 1i: class CNN()")
    print ("-"*80)

    print('Running test on intermediate and final shapes and types')
    _e_char = 50
    _batch_size = 16
    _max_word_length = 25
    _embed_size = 9
    _k = 7
    x_conv_expected_size = [_batch_size, _embed_size, _max_word_length-_k+1]
    x_conv_out_expected_size = [_batch_size, _embed_size]
    _x_reshaped = torch.rand(_batch_size, _e_char, _max_word_length)
    _x_conv_out, _x_conv = CNN(embed_size=_embed_size, k=_k).forward(_x_reshaped)
    assert(list(_x_conv.size()) == x_conv_expected_size), "x_conv shape is incorrect:\n it should be {} but is: {}".format(x_conv_expected_size, list(_x_conv.size()))
    assert(list(_x_conv_out.size()) == x_conv_out_expected_size), "_x_conv_out shape is incorrect:\n it should be {} but is:\n{}".format(x_conv_out_expected_size, list(_x_conv_out.size()))

    print('Running test on intermediate and final values')
    _e_char = 3
    _batch_size = 2
    _max_word_length = 4
    _embed_size = 3
    _k = 3
    x_conv_expected_size = [_batch_size, _embed_size, _max_word_length-_k+1]
    x_conv_out_expected_size = [_batch_size, _embed_size]
    _x_reshaped = torch.rand(_batch_size, _e_char, _max_word_length)
    _model = CNN(embed_size=_embed_size, k=_k, e_char=_e_char)
    _x_conv_out, _x_conv = _model.forward(_x_reshaped)
    print("_x_reshaped:", _x_reshaped)
    print(_model.conv.weight)
    print(_model.conv.bias)
    print("_x_conv:", _x_conv)
    print("_x_conv_out:", _x_conv_out)

    print("All Sanity Checks Passed for Question 1i: class CNN()!")
    print ("-"*80)


if __name__ == '__main__':
    question_1i_sanity_check()"""
### END YOUR CODE

