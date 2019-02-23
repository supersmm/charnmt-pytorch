#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__() 

        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, num_layers=1, bias=True, bidirectional=False)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id), bias=True)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.loss = nn.CrossEntropyLoss(ignore_index = self.target_vocab.char2id['<pad>'])

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        
        #print("input shape (length, batch): ", input.size())
        #print("dec_hidden shape (1, batch, hidden_size) (1, batch, hidden_size): ", dec_hidden[0].size(), dec_hidden[1].size())
        x = self.decoderCharEmb(input)
        output_t, dec_hidden = self.charDecoder(x, dec_hidden)
        scores = self.char_output_projection(output_t)
        #print("scores shape (length, batch, self.vocab_size): ", scores.size())
        return scores, dec_hidden

        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        #print("char_sequence shape (length, batch): ", char_sequence.size())
        scores, dec_hidden = self.forward(char_sequence[:-1, :], dec_hidden)
        #p_t = nn.functional.log_softmax(s_t, dim=-1)
        length, batch, vocab_size = scores.size()
        #print("length, batch, vocab_size: ", length, batch, vocab_size)
        word_loss = self.loss(scores.contiguous().view(length*batch, vocab_size), char_sequence[1:, :].contiguous().view(length*batch,))
        batch_loss = word_loss.sum()

        return batch_loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        decodedWords = []
        batch_size, hidden_size = initialStates[0].size()[1:]
        #print("batch_size, hidden_size: ", batch_size, hidden_size)

        current_chars_id = torch.tensor([[self.target_vocab.start_of_word]*batch_size], device=device)
        output_id = []
        for t in range(max_length):
            #print("t: ", t)
            #print("current_chars_id (length, batch):", current_chars_id.size())
            score, initialStates = self.forward(current_chars_id, initialStates)
            #print("score size (length, batch, self.vocab_size): ", score.size())
            #print("initialStates: ", initialStates)            
            p_t = nn.functional.softmax(score, dim=-1)
            #print("p_t: ", p_t)
            #print("p_t shape (length, batch, self.vocab_size): ", p_t.size())
            Y_t = torch.max(p_t, dim=-1)[1]
            #print("Y_t: ", Y_t)
            #print("Y_t size: ", Y_t.size())
            current_chars_id = Y_t
            output_id.append(current_chars_id.squeeze(dim=0).tolist())
            #print("current_chars_id: ", current_chars_id)
        
        #print("output_id: ", output_id) # (length, batch)
        #print("output_id size:", len(output_id))
        output_np = np.transpose(np.array(output_id)) # (batch, length)
        decodedWords = []
        for b in range(batch_size):
            if self.target_vocab.end_of_word in output_np[b]:
                first_end = list(output_np[b]).index(self.target_vocab.end_of_word)
            else:
                first_end = max_length
            #print("first_end: ", first_end)
            chars = [self.target_vocab.id2char[i] for i in output_np[b]][:first_end]
            #print("chars: ", chars)
            decodedWords.append("".join(chars))
        #print("decodedWords: ", decodedWords)
        return decodedWords

        ### END YOUR CODE

