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
attended2 = Input((4,), name='attended_2')

output = add([
    Dense(units, name='dense_input')(input_),
    Dense(units, use_bias=False, name='dense_in_state')(input_state),
    Dense(units, name='dense_attended')(attended),
    Dense(units, name='dense_attended_2')(attended2),
])

cell = RNNCellModel(
    inputs=input_,
    outputs=output,
    input_states=[input_state],
    output_states=[output],
    constants=[attended, attended2]
)

input_sequence = Input((20, 10))

att_rnn = RNN(cell=cell, return_sequences=True)

output_sequence = att_rnn(input_sequence, constants=[attended, attended2])

model = Model(inputs=[input_sequence, attended, attended2], outputs=output_sequence)

input_sequence_data = np.random.randn(6, 20, 10)
attended_data = np.random.randn(6, 3)
attended2_data = np.random.randn(6, 4)
