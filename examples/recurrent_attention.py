from __future__ import division, print_function

from keras import Input
from keras.engine import Layer


class AttentionCell(Layer):
    pass


class LSTM(Layer):
    pass


class RNN(Layer):
    pass



attended = Input((10, 3))
input_sequence = Input((20, 5))

att_cell = AttentionCell(core_cell=LSTM(32))
att_rnn = RNN(cell=att_cell, return_sequences=True)
output_sequence = att_rnn(input_sequence, constants=attended)
