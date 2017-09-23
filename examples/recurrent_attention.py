from __future__ import division, print_function

import numpy as np

from keras import Input
from keras.engine import Layer
from keras.engine import Model
from keras.layers import Dense, RNN, merge
from keras.layers.recurrent_attention import RNNCellModel

units = 5

input_state = Input((units,))
input_ = Input((10,))

output = merge(
    [Dense(units)(input_), Dense(units, use_bias=False)(input_state)]
)

input_sequence = Input((20, 10))

cell = RNNCellModel(
    inputs=input_,
    outputs=output,
    input_states=[input_state],
    output_states=[output]
)
att_rnn = RNN(cell=cell, return_sequences=True)

output_sequence = att_rnn(input_sequence)

model = Model(inputs=input_sequence, outputs=output_sequence)

input_sequence_data = np.random.randn(3, 20, 10)
