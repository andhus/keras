from __future__ import division, print_function

import numpy as np

from keras import Input
from keras.engine import Layer
from keras.engine import Model
from keras.layers import Dense, RNN, add
from keras.layers.recurrent_attention import RNNCellModel

units = 5

input_state = Input((units,), name='in_state')
input_ = Input((10,), name='input')
attended = Input((3,), name='attended')

output = add([
    Dense(units, name='dense_input')(input_),
    Dense(units, use_bias=False, name='dense_in_state')(input_state),
    Dense(units, name='dense_attended')(attended),
])

cell = RNNCellModel(
    inputs=input_,
    outputs=output,
    input_states=[input_state],
    output_states=[output],
    constants=attended
)

input_sequence = Input((20, 10))

att_rnn = RNN(cell=cell, return_sequences=True)

output_sequence = att_rnn(input_sequence, constants=attended)

model = Model(inputs=[input_sequence, attended], outputs=output_sequence)

input_sequence_data = np.random.randn(6, 20, 10)
attended_data = np.random.randn(6, 3)
