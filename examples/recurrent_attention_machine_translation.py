'''Machine Translation (word-level) with recurrent attention in Keras.

This script demonstrates how to implement basic word-level machine
translation using the recurrent attention mechanism described in [1].

# Summary of the algorithm
The task is to generate a sequence of words in the target language
given the sequence of words in the input language. The overall approach
can be summarized as:
    teacher forcing (i.e. next word is predicted given previous words
    in target language) conditioned on the input language sentence
    using attention.

Main steps:
- Input and target sentences are tokenized (word level).
- Sentences are mapped to word embedding sequences.
- An Encoder encodes the input sequence using a Bidirectional GRU
    into a _sequence_ of encodings - with same length as the input
    sequence.
- A Decoder is constructed using a (unidirectional) GRU with attention:
    At each time step:
    - It is fed the previous target word embedding and previous GRU state.
    - Based on the previous state and the input encoding sequence, it
        computes a weight for each time step (i.e. word index) of the
        input encoding sequence, using a single hidden layer MLP. The
        attention encoding is taken as the weighted sum over the input
        encoding sequence using these weights. (NOTE: this way the
        Decoder has access to all time steps of the input for every
        generated output word.)
    - The attention encoding is concatenated to the previous target word
        embedding and fed as input to the regular GRUCell transform.
    - The output (updated state) of the GRUCell is fed to a single
        hidden (maxout) layer MLP which outputs the probabilities of the
        next target word.
- In inference mode (greedy approach),
    - Encode the input sequence
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the encoded input and 1-char target sequence
        to the decoder and predict probabilities for the next word
    - Select the next word using argmax of probabilities
    - Append the predicted word to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the word limit.

# Data
We use the machine translation dataset described in [2].
To download the data run:
    mkdir -p data/wmt16_mmt
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz
    tar -xf training.tar.gz -C data/wmt16_mmt && rm training.tar.gz
    tar -xf validation.tar.gz -C data/wmt16_mmt && rm validation.tar.gz
    tar -xf mmt16_task1_test.tar.gz -C data/wmt16_mmt && rm mmt16_task1_test.tar.gz

# References
[1] Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/abs/1409.0473
[2] Multi30K: Multilingual English-German Image Descriptions
    https://arxiv.org/abs/1605.00459
    (http://www.statmt.org/wmt16/multimodal-task.html)

# Differences between this implementation and [1]
- A different/older dataset (wmt14) is used in [1].
- The Tokenization here is similar but not identical to [1].
- Initialisation of weights are not identical here and [1].
- Normalisation of gradients is done when L2-norm > 1 in [1].
- In the detailed description of the architecture in [1] it is stated:
    "From here on, we omit all bias terms in order to increase
    readability". It is thus not fully clear which linear transformations
    also has bias terms, here all have.
- TODO(2) Beam search is used in inference in [1]. This would be nice to
    add to make example complete.
- TODO(3) A "cascading architecture" is use in [1] by feeding not only
    the GRU output to the readout layer but also the attention encoding.
    This can be done by:
    1) concatenating the attention encoding to the output of the
        wrapped cell in the _RNNAttentionCell. However, then it must be
        concatenated to the wrapped cell state as well which causes
        cognitive overhead. This must be done because `state_size` and
        `output_size` of an RNNCell must be the equal for masking to
        work currently (this is a separate issue that can and should be
        fixed though).
    2) Adding "return_state_sequences" to RNN - this way concatenation
        can be done externally and if also offers much more flexibility
        for inspecting "what is attended" by the attention mechanism.
(- There is no mention of Dropout or other regularisation methods in [1],
    this could improve performance.)
'''

from __future__ import print_function

import os
import numpy as np

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.base_layer import _collect_previous_mask
from keras.layers import Layer, InputSpec
from keras.layers import Input, Embedding, Bidirectional, RNN, GRU, GRUCell
from keras.layers import TimeDistributed, Dense, concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import has_arg


class _RNNAttentionCell(Layer):
    """Base class for recurrent attention mechanisms.

    This base class implements the RNN cell interface and defines a standard
    way for attention mechanisms to interact with a (wrapped) "core" RNN cell
    (such as the `SimpleRNNCell`, `GRUCell` or `LSTMCell`).

    The main idea is that the attention mechanism, implemented by
    `attention_call` in extensions of this class, computes an "attention
    encoding", based on the attended input as well as the input and the core
    cell state(s) at the current time step, which will be used as modified
    input for the core cell.

    # Arguments
        cell: A RNN cell instance. The cell to wrap by the attention mechanism.
            See docs of `cell` argument in the `RNN` Layer for further details.
        attend_after: Boolean (default False). If True, the attention
            transformation defined by `attention_call` will be applied after
            the core cell transformation (and the attention encoding will be
            used as input for core cell transformation next time step).
        concatenate_input: Boolean (default True). If True the concatenation of
            the attention encoding and the original input will be used as input
            for the core cell transformation. If set to False, only the
            attention encoding will be used as input for the core cell
            transformation.

    # Abstract Methods and Properties
        Extension of this class must implement:
            - `attention_build` (method): Builds the attention transformation
              based on input shapes.
            - `attention_call` (method): Defines the attention transformation
              returning the attention encoding.
            - `attention_size` (property): After `attention_build` has been
              called, this property should return the size (int) of the
              attention encoding. Do this by setting `_attention_size` in scope
              of `attention_build` or by implementing `attention_size`
              property.
        Extension of this class can optionally implement:
            - `attention_state_size` (property): Default [`attention_size`].
              If the attention mechanism has it own internal states (besides
              the attention encoding which is by default the only part of
              `attention_states`) override this property accordingly.
        See docs of the respective method/property for further details.

    # Details of interaction between attention and cell transformations
        Let "cell" denote core (wrapped) RNN cell and "att(cell)" the complete
        attentive RNN cell defined by this class. We write the core cell
        transformation as:

            y{t}, s_cell{t+1} = cell.call(x{t}, s_cell{t})

        where y{t} denotes the output, x{t} the input at and s_cell{t} the core
        cell state(s) at time t and s_cell{t+1} the updated state(s).

        We can then write the complete "attentive" cell transformation as:

            y{t}, s_att(cell){t+1} = att(cell).call(x{t}, s_att(cell){t},
                                                    constants=attended)

        where s_att(cell) denotes the complete states of the attentive cell,
        which consists of the core cell state(s) followed but the attention
        state(s), and attended denotes the tensor attended to (note: no time
        indexing as this is the same constant input at each time step).

        Internally, this is how the attention transformation, implemented by
        `attention_call`, interacts with the core cell transformation
        `cell.call`:

        - with `attend_after=False` (default):
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t},
                                                        attended, s_att{t})
            with `concatenate_input=True` (default):
                x'{t} = [x{t}, a{t}]
            else:
                x'{t} = a{t}
            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})

        - with `attend_after=True`:
            with `concatenate_input=True` (default):
                x'{t} = [x{t}, a{t-1}]
            else:
                x'{t} = a{t-1}
            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t+1},
                                                        attended, s_att{t})

        where a{t} denotes the attention encoding, s_att{t} the attention
        state(s), x'{t} the modified core cell input and [x{.}, a{.}] the
        (tensor) concatenation of the input and attention encoding.
    """

    def __init__(self, cell,
                 attend_after=False,
                 concatenate_input=False,
                 **kwargs):
        self.cell = cell  # must be set before calling super
        super(_RNNAttentionCell, self).__init__(**kwargs)
        self.attend_after = attend_after
        self.concatenate_input = concatenate_input
        self.attended_spec = None
        self._attention_size = None

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        """The main logic for computing the attention encoding.

        # Arguments
            inputs: The input at current time step.
            cell_states: States for the core RNN cell.
            attended: The constant tensor(s) to attend at each time step.
            attention_states: States dedicated for the attention mechanism.
            attended_mask: Collected masks for the attended.
            training: Whether run in training mode or not.

        # Returns
            attention_h: The computed attention encoding at current time step.
            attention_states: States to be passed to next `attention_call`. By
                default this should be [`attention_h`].
                NOTE: if additional states are used, these should be appended
                after `attention_h`, i.e. `attention_states[0]` should always
                be `attention_h`.
        """
        raise NotImplementedError(
            '`attention_call` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        """Build the attention mechanism.

        NOTE: `self._attention_size` should be set in this method to the size
        of the attention encoding (i.e. size of first `attention_states`)
        unless `attention_size` property is implemented in another way.

        # Arguments
            input_shape: Tuple of integers. Shape of the input at a single time
                step.
            cell_state_size: List of tuple of integers.
            attended_shape: List of tuple of integers.

            NOTE: both `cell_state_size` and `attended_shape` will always be
            lists - for simplicity. For example: even if (wrapped)
            `cell.state_size` is an integer, `cell_state_size` will be a list
            of this one element.
        """
        raise NotImplementedError(
            '`attention_build` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    @property
    def attention_size(self):
        """Size off attention encoding, an integer.
        """
        if self._attention_size is None and self.built:
            raise NotImplementedError(
                'extensions of `{}` must either set property `_attention_size`'
                ' in `attention_build` or implement the or implement'
                ' `attention_size` in some other way'.format(
                    self.__class__.__name__))

        return self._attention_size

    @property
    def attention_state_size(self):
        """Size of attention states, defaults to `attention_size`, an integer.

        Modify this property to return list of integers if the attention
        mechanism has several internal states. Note that the first size should
        always be the size of the attention encoding, i.e.:
            `attention_state_size[0]` = `attention_size`
        """
        return self.attention_size

    @property
    def state_size(self):
        """Size of states of the complete attentive cell, a tuple of integers.

        The attentive cell's states consists of the core RNN cell state size(s)
        followed by attention state size(s). NOTE it is important that the core
        cell states are first as the first state of any RNN cell should be same
        as the cell's output.
        """
        state_size_s = []
        for state_size in [self.cell.state_size, self.attention_state_size]:
            if hasattr(state_size, '__len__'):
                state_size_s += list(state_size)
            else:
                state_size_s.append(state_size)

        return tuple(state_size_s)

    def call(self, inputs, states, constants, training=None):
        """Complete attentive cell transformation.
        """
        attended = constants
        attended_mask = _collect_previous_mask(attended)
        # attended and mask are always lists for uniformity:
        if not isinstance(attended_mask, list):
            attended_mask = [attended_mask]
        cell_states = states[:self._num_wrapped_states]
        attention_states = states[self._num_wrapped_states:]

        if self.attend_after:
            call = self.call_attend_after
        else:
            call = self.call_attend_before

        return call(inputs=inputs,
                    cell_states=cell_states,
                    attended=attended,
                    attention_states=attention_states,
                    attended_mask=attended_mask,
                    training=training)

    def call_attend_before(self,
                           inputs,
                           cell_states,
                           attended,
                           attention_states,
                           attended_mask,
                           training=None):
        """Complete attentive cell transformation, if `attend_after=False`.
        """
        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=cell_states,
            attended=attended,
            attention_states=attention_states,
            attended_mask=attended_mask,
            training=training)

        if self.concatenate_input:
            cell_input = concatenate([attention_h, inputs])
        else:
            cell_input = attention_h

        if has_arg(self.cell.call, 'training'):
            output, new_cell_states = self.cell.call(cell_input, cell_states,
                                                     training=training)
        else:
            output, new_cell_states = self.cell.call(cell_input, cell_states)

        return output, new_cell_states + new_attention_states

    def call_attend_after(self,
                          inputs,
                          cell_states,
                          attended,
                          attention_states,
                          attended_mask,
                          training=None):
        """Complete attentive cell transformation, if `attend_after=True`.
        """
        attention_h_previous = attention_states[0]

        if self.concatenate_input:
            cell_input = concatenate([attention_h_previous, inputs])
        else:
            cell_input = attention_h_previous

        if has_arg(self.cell.call, 'training'):
            output, new_cell_states = self.cell.call(cell_input, cell_states,
                                                     training=training)
        else:
            output, new_cell_states = self.cell.call(cell_input, cell_states)

        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=new_cell_states,
            attended=attended,
            attention_states=attention_states,
            attended_mask=attended_mask,
            training=training)

        return output, new_cell_states, new_attention_states

    @staticmethod
    def _num_elements(x):
        if hasattr(x, '__len__'):
            return len(x)
        else:
            return 1

    @property
    def _num_wrapped_states(self):
        return self._num_elements(self.cell.state_size)

    @property
    def _num_attention_states(self):
        return self._num_elements(self.attention_state_size)

    def build(self, input_shape):
        """Builds attention mechanism and wrapped cell (if keras layer).

        Arguments:
            input_shape: list of tuples of integers, the input feature shape
                (inputs sequence shape without time dimension) followed by
                constants (i.e. attended) shapes.
        """
        if not isinstance(input_shape, list):
            raise ValueError('input shape should contain shape of both cell '
                             'inputs and constants (attended)')

        attended_shape = input_shape[1:]
        input_shape = input_shape[0]
        self.attended_spec = [InputSpec(shape=shape) for shape in attended_shape]
        if isinstance(self.cell.state_size, int):
            cell_state_size = [self.cell.state_size]
        else:
            cell_state_size = list(self.cell.state_size)
        self.attention_build(
            input_shape=input_shape,
            cell_state_size=cell_state_size,
            attended_shape=attended_shape,
        )

        if isinstance(self.cell, Layer):
            cell_input_shape = (input_shape[0],
                                self.attention_size +
                                input_shape[-1] if self.concatenate_input
                                else self._attention_size)
            self.cell.build(cell_input_shape)

        self.built = True

    def compute_output_shape(self, input_shape):
        if hasattr(self.cell.state_size, '__len__'):
            cell_output_dim = self.cell.state_size[0]
        else:
            cell_output_dim = self.cell.state_size

        return input_shape[0], cell_output_dim

    @property
    def trainable_weights(self):
        return (super(_RNNAttentionCell, self).trainable_weights +
                self.cell.trainable_weights)

    @property
    def non_trainable_weights(self):
        return (super(_RNNAttentionCell, self).non_trainable_weights +
                self.cell.non_trainable_weights)

    def get_config(self):
        config = {'attend_after': self.attend_after,
                  'concatenate_input': self.concatenate_input}

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(_RNNAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseAnnotationAttention(_RNNAttentionCell):
    """Recurrent attention mechanism for attending sequences.

    This class implements the attention mechanism used in [1] for machine
    translation. It is, however, a generic sequence attention mechanism that can be
    used for other sequence-to-sequence problems.

    As any recurrent attention mechanism extending `_RNNAttentionCell`, this class
    should be used in conjunction with a core (non attentive) RNN Cell, such as the
    `SimpleRNNCell`, `LSTMCell` or `GRUCell`. It modifies the input of the core cell
    by attending to a constant sequence (i.e. independent of the time step of the
    recurrent application of the attention mechanism). The attention encoding is
    computed  by using a single hidden layer MLP which computes a weighting over the
    attended input. The MLP is applied to each time step of the attended, together
    with the previous state. The attention encoding is the taken as the weighted sum
    over the attended input using these weights.

    # Arguments
        cell: A RNN cell instance. The core RNN cell wrapped by this attention
            mechanism. See docs of `cell` argument in the `RNN` Layer for further
            details.
        units: the number of hidden units in the single hidden MLP used for
            computing the attention weights.
        kernel_initializer: Initializer for all weights matrices
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for all bias vectors
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            all weights matrices. (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to all biases
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            all weights matrices. (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to all bias vectors
            (see [constraints](../constraints.md)).

     # Examples

    ```python
        # machine translation (similar to the architecture used in [1])
        x = Input((None,), name="input_sequences")
        y = Input((None,), name="target_sequences")
        x_emb = Embedding(INPUT_NUM_WORDS, 256, mask_zero=True)(x)
        y_emb = Embedding(TARGET_NUM_WORDS, 256, mask_zero=True)(y)
        encoder = Bidirectional(GRU(512, return_sequences=True))
        x_enc = encoder(x_emb)
        decoder = RNN(cell=DenseAnnotationAttention(cell=GRUCell(512), units=128),
                      return_sequences=True)
        h = decoder(y_emb, constants=x_enc)
        y_pred = TimeDistributed(Dense(TARGET_NUM_WORDS, activation='softmax'))(h)
        model = Model([y, x], y_pred)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER)
    ```

    # Details of attention mechanism
    Let {attended_1, ..., attended_I} denote the attended input sequence, where
    attended_i is the i:t attended input vector, h_cell_tm1 the previous state of
    the core cell at the recurrent time step t. Then the attention encoding at
    time step t is computed as follows:

        e_i = MLP([attended_i, h_cell_tm1])  # [., .] denoting concatenation
        a_i = softmax_i({e_1, ..., e_I})
        attention_h_t = sum_i(a_i * h_i)

    # References
    [1] Neural Machine Translation by Jointly Learning to Align and Translate
        https://arxiv.org/abs/1409.0473
    """
    def __init__(self, cell,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseAnnotationAttention, self).__init__(cell, **kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        # only one attended sequence (verified in build)
        [attended, u] = attended
        attended_mask = attended_mask[0]
        h_cell_tm1 = cell_states[0]

        # compute attention weights
        w = K.repeat(K.dot(h_cell_tm1, self.W_a) + self.b_UW, K.shape(attended)[1])
        e = K.exp(K.dot(K.tanh(w + u), self.v_a) + self.b_v)

        if attended_mask is not None:
            e = e * K.cast(K.expand_dims(attended_mask, -1), K.dtype(e))

        # weighted average of attended
        a = e / K.sum(e, axis=1, keepdims=True)
        c = K.sum(a * attended, axis=1, keepdims=False)

        return c, [c]

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 2:
            raise ValueError('There must be two attended tensors')
        for a in attended_shape:
            if not len(a) == 3:
                raise ValueError('only support attending tensors with dim=3')
        [attended_shape, u_shape] = attended_shape

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1]
        units = u_shape[-1]

        kernel_kwargs = dict(initializer=self.kernel_initializer,
                             regularizer=self.kernel_regularizer,
                             constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(cell_state_size[0], units),
                                   name='W_a', **kernel_kwargs)
        self.v_a = self.add_weight(shape=(units, 1),
                                   name='v_a', **kernel_kwargs)

        bias_kwargs = dict(initializer=self.bias_initializer,
                           regularizer=self.bias_regularizer,
                           constraint=self.bias_constraint)
        self.b_UW = self.add_weight(shape=(units,),
                                    name="b_UW", **bias_kwargs)
        self.b_v = self.add_weight(shape=(1,),
                                   name="b_v", **bias_kwargs)

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseAnnotationAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    # DATA_DIR = 'data/wmt16_mmt'
    DATA_DIR = '/Users/andershuss/Datasets/WMT16/mmt'

    FROM_LANGUAGE = 'en'
    TO_LANGUAGE = 'de'

    # Meta parameters
    MAX_UNIQUE_WORDS = 30000
    MAX_WORDS_PER_SENTENCE = 40  # inf in [1]
    EMBEDDING_SIZE = 620  # `m` in [1]
    RECURRENT_UNITS = 1000  # `n` in [1]
    DENSE_ATTENTION_UNITS = 1000  # fixed equal to `n` in [1]
    READOUT_HIDDEN_UNITS = 500  # `l` in [1]
    OPTIMIZER = Adadelta(rho=0.95, epsilon=1e-6)
    BATCH_SIZE = 80
    EPOCHS = 5

    # Load and tokenize the data
    start_token = "'start'"
    end_token = "'end'"
    # NOTE: using single quotes (which are not dropped by Tokenizer by default)
    # for the tokens to be distinguished from other use of "start" and "end"

    def get_sentences(partion, language):
        fpath = os.path.join(DATA_DIR, partion + '.' + language)
        with open(fpath, 'r') as f:
            sentences = f.readlines()
        return ["{} {} {}".format(start_token, sentence, end_token)
                for sentence in sentences]

    input_texts_train = get_sentences("train", FROM_LANGUAGE)
    input_texts_val = get_sentences("val", FROM_LANGUAGE)
    target_texts_train = get_sentences("train", TO_LANGUAGE)
    target_texts_val = get_sentences("val", TO_LANGUAGE)

    input_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
    target_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
    input_tokenizer.fit_on_texts(input_texts_train + input_texts_val)
    target_tokenizer.fit_on_texts(target_texts_train + target_texts_val)

    input_seqs_train = input_tokenizer.texts_to_sequences(input_texts_train)
    input_seqs_val = input_tokenizer.texts_to_sequences(input_texts_val)
    target_seqs_train = target_tokenizer.texts_to_sequences(target_texts_train)
    target_seqs_val = target_tokenizer.texts_to_sequences(target_texts_val)

    input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val = (
        pad_sequences(seq, maxlen=MAX_WORDS_PER_SENTENCE, padding='post')
        for seq in [input_seqs_train,
                    input_seqs_val,
                    target_seqs_train,
                    target_seqs_val]
    )

    # Build the model
    x = Input((None,), name="input_sequences")
    y = Input((None,), name="target_sequences")
    x_emb = Embedding(input_tokenizer.num_words,
                      EMBEDDING_SIZE,
                      mask_zero=True)(x)
    y_emb = Embedding(target_tokenizer.num_words,
                      EMBEDDING_SIZE,
                      mask_zero=True)(y)

    encoder = Bidirectional(GRU(RECURRENT_UNITS, return_sequences=True))
    x_enc = encoder(x_emb)

    cell = DenseAnnotationAttention(cell=GRUCell(RECURRENT_UNITS))
    decoder = RNN(cell=cell, return_sequences=True)
    u = TimeDistributed(Dense(DENSE_ATTENTION_UNITS, use_bias=False))(x_enc)
    h1 = decoder(y_emb, constants=[x_enc, u])

    def dense_maxout(x_):
        """Implements a dense maxout layer where max is taken
        over _two_ units"""
        x_ = Dense(READOUT_HIDDEN_UNITS * 2)(x_)
        x_1 = x_[:, :READOUT_HIDDEN_UNITS]
        x_2 = x_[:, READOUT_HIDDEN_UNITS:]
        return K.max(K.stack([x_1, x_2], axis=-1), axis=-1, keepdims=False)

    maxout_layer = TimeDistributed(Lambda(dense_maxout))
    h2 = maxout_layer(concatenate([h1, y_emb]))
    output_layer = TimeDistributed(
        Dense(target_tokenizer.num_words, activation='softmax'))
    y_pred = output_layer(h2)

    model = Model([y, x], y_pred)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER)

    # reduce data TODO temp
    # input_seqs_train = input_seqs_train[:80]
    # target_seqs_train = target_seqs_train[:80]
    # input_seqs_val = input_seqs_val[:80]
    # target_seqs_val = target_seqs_val[:80]
    #
    #
    # model.fit([target_seqs_train[:, :-1], input_seqs_train],
    #           target_seqs_train[:, 1:, None],
    #           batch_size=BATCH_SIZE,
    #           epochs=EPOCHS,
    #           validation_data=(
    #               [target_seqs_val[:, :-1], input_seqs_val],
    #               target_seqs_val[:, 1:, None]))

    # TODO add logic for greedy/beam search generation

    # TODO!!
    # [/] graph construction simpler -> seems to be required to do this way
    # [/] start and end token for Tokenizer, must be changed (org PR)
    # [-] Beam search

    # pre-processing (once per input sequence)
    prep_model = Model(x, [x_enc, u])

    # decoder to output with exposed state
    x_enc_new = Input(batch_shape=K.int_shape(x_enc))
    u_new = Input(batch_shape=K.int_shape(u))

    initial_state = [Input((size,)) for size in cell.state_size]
    decoder_new = RNN(cell=cell, return_sequences=True, return_state=True)
    h1_new_and_state = decoder_new(y_emb,
                                   initial_state=initial_state,
                                   constants=[x_enc_new, u_new])
    h1_new = h1_new_and_state[0]
    state = h1_new_and_state[1:]
    h2_new = maxout_layer(concatenate([h1_new, y_emb]))
    y_pred_new = output_layer(h2_new)
    decoder_output_model = Model([y, x_enc_new, u_new] + initial_state, [y_pred_new] + state)

    def translate_greedy(input_text, init_text=None, t_max=100):
        """

        :param input_text:
        :param init_text:
        :param t_max:
        :return:
        """
        if init_text is None:
            init_text = start_token
        t = 0
        y_t = np.array(target_tokenizer.texts_to_sequences([init_text]))
        y_0_to_t = [y_t]
        x = input_tokenizer.texts_to_sequences([input_text])
        state_t = [np.zeros((1, size)) for size in cell.state_size]
        [x_enc, u] = prep_model.predict(x)
        end_idx = target_tokenizer.word_index[end_token]
        while y_t[0, 0] != end_idx and t < t_max:
            outputs = decoder_output_model.predict([y_t, x_enc, u] + state_t)
            t += 1
            state_t = outputs[1:]
            y_pred = outputs[0]
            y_t = np.argmax(y_pred, axis=-1)
            y_0_to_t.append(y_t)
        y = np.hstack(y_0_to_t)
        output_text = target_tokenizer.sequences_to_texts(y)[0]

        return output_text

    def translate_beam_serach(input_text,
                              init_text=None,
                              t_max=None,
                              search_width=50):
        def argmaxk(a, k):
            return np.argpartition(a, -k)[-k:]

        if init_text is None:
            init_text = start_token
        t = 0
        y_0 = np.array(target_tokenizer.texts_to_sequences([init_text]))[0]
        x = input_tokenizer.texts_to_sequences([input_text])
        state_0 = [np.zeros((size,)) for size in cell.state_size]
        [x_enc, u] = prep_model.predict(x)
        end_idx = target_tokenizer.word_index[end_token]

        import heapq

        complete = []
        incomplete = [{"score": 0., "y_0:t": [y_0], "state": state_0}]

        while len(complete) < search_width and t < t_max:
            t += 1
            y_tm1 = np.vstack([beam["y_0:t"][-1] for beam in incomplete])
            state_tm1 = [
                np.vstack([beam["state"][s] for beam in incomplete])
                for s in range(len(state_0))
            ]
            outputs = decoder_output_model.predict([y_tm1, x_enc, u] + state_tm1)
            state_t = outputs[1:]
            y_pred = outputs[0]
            incomplete_new = []
            for i, beam in enumerate(incomplete):
                max_idxs = argmaxk(y_pred[i], search_width)
                l = len(beam["y_0:t"]) - 1  # don't count 'start' token
                for idx in max_idxs:
                    proba = y_pred[i][idx]
                    new_score = (beam["score"] * l + np.log(proba)) / (l + 1)
                    if (len(incomplete_new) < search_width or
                            new_score > incomplete_new[0] or
                            idx == end_idx):
                        # add this version
                        beam_new = beam.copy()
                        beam_new["score"] = new_score
                        beam_new["y_0:t"] = beam["y_0:t"] + [np.array([idx])]
                        beam_new["state"] = state_t
                        if idx == end_idx:
                            heapq.heappush(complete, beam_new)
                        elif len(incomplete_new) < search_width:
                            heapq.heappush(incomplete_new, beam_new)
                        else:
                            heapq.heapreplace(incomplete_new, beam_new)
                    else:
                        break
            incomplete = incomplete_new

            for beam in complete + incomplete:
                beam["text"] = target_tokenizer.sequences_to_texts(
                    np.concatenate(beam["y_0:t"]))

            return complete, incomplete
