from __future__ import division, print_function

import random

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, TimeDistributed, LSTMCell, RNN, LSTM, concatenate, Embedding, Bidirectional

from keras.layers.attention import MixtureOfGaussian1DAttention


n_characters = 28
xy_data = np.zeros((200, 100, 2))
text_data = np.zeros((200, 10, n_characters))


# handwriting generation, conditioned on text to be generated

xy = Input((None, 2))  # coordinates of handwriting trajectory
text = Input((None, n_characters))  # sequence of characters to be generated

cell = MixtureOfGaussian1DAttention(LSTMCell(64), components=3, heads=2)
attention_lstm = RNN(cell, return_sequences=True)
h = attention_lstm(xy, constants=text)
xy_pred = TimeDistributed(Dense(2, activation=None))(h)

model = Model(inputs=[xy, text], outputs=xy_pred)
model.compile(optimizer='adam', loss='mse')
model.fit([xy_data[:, :-1], text_data], xy_data[:, 1:])


# machine translation
n_input_tokens = 200
n_target_tokens = 200

input_sentence = Input((None,))  # sequence of words language A
target_sentence = Input((None,))  # sequence of words language B

input_embeddings = Embedding(n_input_tokens, 620)(input_sentence)
target_embeddings = Embedding(n_target_tokens, 620)(target_sentence)

input_encoding = Bidirectional(LSTM(1000))(input_embeddings)

h = RNN(
    DenseAnnotationAttention(LSTMCell(1000)),
    return_sequences=True
)(target_sentence, constants=input_encoding)

target_sentence_pred = TimeDistributed(Dense(n_target_tokens, activation=None))(h)
# NOTE "maxout" is used in the paper


# stacked cascading version with forwarding of attention encoding to subsequent layers

cell = MixtureOfGaussian1DAttention(LSTMCell(64), components=3, heads=2)
attention_lstm = RNN(cell, return_state_sequences=True)  # currently not supported
state_seqs = attention_lstm(xy, constants=text)
# states of the attentive cell is states of wrapped cell (h, c)
# followed by attention encoding and any additional states of attention mechanism
h0_lstm = state_seqs[0]
h0_att = state_seqs[2]
h1 = RNN(LSTMCell(64), return_sequences=True)(concatenate([h0_lstm, h0_att]))
h2 = RNN(LSTMCell(64), return_sequences=True)(concatenate([h1, h0_lstm, h0_att]))
xy_pred = TimeDistributed(Dense(2, activation=None))(h2)


# alt. as a new layer or Model subclassing API
attention_lstm = MixtureOfGaussian1DAttentionRNN(
    cell=LSTMCell(64),
    components=3,
    heads=2,
    return_sequences=True  # here return_state_sequences could be supported or other
)
h = attention_lstm(xy, attended=text)

# alt. as a new layer or Model subclassing API
attention_lstm = AttentionRNN(
    cell=LSTMCell(64),
    attention_mechanism=MixtureOfGaussian1DAttention(components=3, heads=2),
    components=3,
    heads=2,
    return_sequences=True  # here return_state_sequences could be supported or other
)
h = attention_lstm(xy, attended=text)



# FF Attention: Transformer
n_input_tokens = 4
n_target_tokens = 4

input_sequence = Input((None, 1))
target_sequence_tm1 = Input((None, 1))
transformer = Tranformer(
    input_tokens=n_input_tokens,  # must be provided if embeddings created internally
    target_tokens=n_target_tokens,  # must be provided if embeddings created internally
    units_model=512,
    units_ff=2048,
    units_keys=64,
    units_values=64,
    layers=6,
    heads=3
)
h = transformer([input_sequence, target_sequence_tm1])
target_sequence_pred = TimeDistributed(Dense(n_target_tokens, activation=None))



# alternative with functional API for defining the Attentive cell

# define complete attentive cell using functional API
units = 32
xy_t = Input((2,))
text = Input((None, n_characters))  # note that this is a sequence
h_tm1 = Input((units,))
c_tm1 = Input((units,))
h_att_t = AttentionMechanism(inputs=concatenate(xy_t, h_tm1), attended=text)
x_t = concatenate([xy_t, h_att_t])
h_t, c_t = LSTMCell(units)(inputs=x_t, states=[h_tm1, c_tm1])
cell = CellModel(  # creates a valid cell implementation based on functional definition
    inputs=xy_t,
    outputs=h,
    input_states=[h_tm1, c_tm1],
    output_states=[h_t, c_t],
    constants=text
)