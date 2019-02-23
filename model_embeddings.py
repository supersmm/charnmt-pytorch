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

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        # print("vocab: ", vocab, "len(vocab): ", len(vocab))
        self.embed_size = embed_size # e_word
        self.e_char = 50
        self.dropout_rate = 0.3
        self.embeddings = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx=pad_token_idx)
        self.cnn = CNN(embed_size=self.embed_size)
        self.highway = Highway(embed_size=self.embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate, inplace=False)


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary ## x_padded

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch ## x_word_emb
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x_padded = input
        sentence_length, batch_size, max_word_length = x_padded.size()
        # print("x_padded shape (sentence_length, batch_size, max_word_length): ", x_padded.size())
        x_char_emb = self.embeddings(x_padded) # (sentence_length, batch_size, max_word_length, e_char)
        # print("x_char_emb shape (sentence_length, batch_size, max_word_length, e_char): ", x_char_emb.size())
        x_reshaped = x_char_emb.view(-1, max_word_length, self.e_char).permute(0, 2, 1).contiguous() # (sentence_length*batch_size, e_char, max_word_length)
        # print("x_reshaped shape (sentence_length*batch_size, e_char, max_word_length): ", x_reshaped.size())
        x_conv_out = self.cnn(x_reshaped)
        # print("x_conv_out shape (batch_size*sentence_length, embed_size): ", x_conv_out.size())
        x_highway = self.highway(x_conv_out)
        # print("x_highway shape (batch_size*sentence_length, embed_size): ", x_highway.size())
        x_word_emb = self.dropout(x_highway)
        # e_word = x_highway.size()[1]
        x_word_emb = x_word_emb.view(sentence_length, batch_size, self.embed_size)
        # print("x_word_emb shape: ", x_word_emb.size())

        return x_word_emb

        ### END YOUR CODE

