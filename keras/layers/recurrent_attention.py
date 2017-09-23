# -*- coding: utf-8 -*-
from __future__ import absolute_import

import abc

from keras.engine import Model
from keras.layers import Wrapper, concatenate
from keras import backend as K


class AttentionCellABC(object):

    def call(self, inputs, states, constants):
        """
        # Returns
            inputs: input tensor
            states: (list of) state tensor(s)
            constants: (list of) attended tensor(s)
        """
        pass

    @abc.abstractproperty
    def state_size(self):
        pass


class RNNCellModel(Model):

    def __init__(
        self,
        inputs,
        outputs,
        input_states,
        output_states,
        constants=None
    ):
        input_states = to_list(input_states)
        constants = to_list(constants) if constants else None
        self._n_states = len(input_states)
        self._n_constants = len(constants) if constants else 0
        super(RNNCellModel, self).__init__(
            inputs=self._get_model_inputs(inputs, input_states, constants),
            outputs=self._get_model_outputs(outputs, output_states)
        )
        states_shape = get_shape(input_states)
        if not states_shape == get_shape(output_states):
            raise ValueError(
                'shape of input states must same as shape of output states'
            )
        self._state_size = [state_shape[-1] for state_shape in states_shape]

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, states, constants=None, training=None):
        output, states = self._get_output_and_output_states(
            super(RNNCellModel, self).call(
                self._get_model_inputs(inputs, states, constants)
            )
        )
        # TOdO use training?
        return output, states

    def _get_model_inputs(self, inputs, input_states, constants):
        return [inputs] + list(input_states) + (constants or [])

    def _get_model_outputs(self, outputs, output_states):
        return [outputs] + output_states

    def _get_output_and_output_states(self, outputs):
        return outputs[0], outputs[1:]


class AttentionCellBase(Wrapper):

    def __init__(
        self,
        units,
        wrapped_cell,
        attend_after=False,
        concatenate_input=False,
        return_attention=False,
        **kwargs
    ):
        super(AttentionCellABC, self).__init__(layer=wrapped_cell, **kwargs)
        self.units = units
        self.attend_after = attend_after
        self.concatenate_input = concatenate_input
        self.return_attention = return_attention

        # set either in call or by setting attended property
        self._attended = None

    @abc.abstractproperty
    def attention_state_size(self):
        pass

    @abc.abstractmethod
    def attention_call(
        self,
        attended,
        attention_states,
        inputs,
        wrapped_cell_states,
    ):
        """This method implements the core logic for computing the attention
        representation.

        # Arguments
            attended: the same tensor at each timestep
            attention_states: states from previous attention step, by
                default attention from last step but can be extended
            inputs: the input at current timesteps
            wrapped_cell_states: states for recurrent layer (excluding constants
                like dropout tensors) from previous state if attend_after=False
                otherwise from current time step.

        # Returns
            attention_h: the computed attention representation at current
                timestep
            attention_states: states to be passed to next attention_step, by
                default this is just [attention_h]. NOTE if more states are
                used, these should be _appended_ to attention states,
                attention_states[0] should always be attention_h.
        """
        pass

    @abc.abstractmethod
    def attention_build(
        self,
        attended_shape,
        input_shape,
        wrapped_cell_state_size
    ):
        pass

    @property
    def attended(self):
        if self._attended is None:
            raise AttributeError('attended has not been set')
        return self._attended

    @attended.setter
    def attended(self, attended):
        if self._attended is not None:
            raise RuntimeError('attended should only be set once')
        self._attended = attended

    @property
    def wrapped_cell(self):
        return self.layer

    @property
    def state_size(self):
        """
        # Returns
            tuple of attention state size(s) followed by wrapped cell state
            size(s)
        """
        state_size_s = []
        for state_size in [
            self.wrapped_cell.state_size,  # these first to comply with states[0] beeing output by default
            self.attention_state_size
        ]:
            if hasattr(state_size, '__len__'):
                state_size += list(state_size)
            else:
                state_size.append(state_size)

        return tuple(state_size_s)

    def call(self, inputs, states, constants=None):
        attended = constants or self.attended
        if attended is None:
            raise RuntimeError(
                'attended must either be passed in call or set as property'
            )
        state_components = self.get_states_components(states)
        if self.attend_after:
            return self.call_attend_after(inputs, attended, *state_components)
        else:
            return self.call_attend_before(inputs, attended, *state_components)

    def call_attend_before(
        self,
        inputs,
        attended,
        attention_states_tm1,
        wrapped_cell_states_tm1,
        wrapped_cell_constants
    ):
        attention_h, attention_states = \
            self.attention_call(
                attended=attended,
                attention_states=attention_states_tm1,
                inputs=inputs,
                wrapped_cell_states=wrapped_cell_states_tm1,
            )
        if self.concatenate_input:
            wrapped_cell_input = concatenate([attention_h, inputs])
        else:
            wrapped_cell_input = attention_h

        output, recurrent_states = self.wrapped_cell.call(
            wrapped_cell_input,
            wrapped_cell_states_tm1 + wrapped_cell_constants
        )

        if self.return_attention:
            output = concatenate([output, attention_h])

        return output, recurrent_states + attention_states

    def call_attend_after(
        self,
        inputs,
        attended,
        attention_states_tm1,
        wrapped_cell_states_tm1,
        wrapped_cell_constants
    ):
        attention_h_tm1 = attention_states_tm1[0]

        if self.concatenate_input:
            wrapped_cell_input = concatenate([attention_h_tm1, inputs])
        else:
            wrapped_cell_input = attention_h_tm1

        output, wrapped_cell_states = self.wrapped_cell.call(
            wrapped_cell_input,
            wrapped_cell_states_tm1 + wrapped_cell_constants
        )

        attention_h, attention_states = \
            self.attention_call(
                attended=attended,
                attention_states=attention_states_tm1,
                inputs=inputs,
                wrapped_cell_states=wrapped_cell_states
            )

        if self.return_attention:
            output = concatenate([output, attention_h])

        return output, wrapped_cell_states, attention_states

    def get_states_components(self, states):
        nws = self._n_wrapped_states
        nas = self._n_attention_states
        attention_states = states[:nws]
        wrapped_cell_states = states[nws:nws + nas]
        wrapped_cell_constants = states[nws + nas:]  # TODO remove?

        return attention_states, wrapped_cell_states, wrapped_cell_constants

    @property
    def _n_wrapped_states(self):
        if hasattr(self.wrapped_cell.state_size, '__len__'):
            return len(self.wrapped_cell.state_size)
        else:
            return 1

    @property
    def _n_attention_states(self):
        if hasattr(self.attention_state_size, '__len__'):
            return len(self.attention_state_size)
        else:
            return 1

    def build(self, input_shape):
        """step input shape"""
        if self.attended is None:
            raise RuntimeError('attended must be set before build is called')

        self.attention_build(
            self._get_attended_shape(),
            input_shape,
            self.wrapped_cell.state_size
        )
        wrapped_cell_input_shape = (
            input_shape[0],
            self.units + input_shape[-1]
            if self.concatenate_input else self.units
        )
        # TODO check that wrapped is a layer (?)
        self.wrapped_cell.build(wrapped_cell_input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        if hasattr(self.wrapped_cell.state_size, '__len__'):
            wrapped_output_dim = self.wrapped_cell.state_size[0]
        else:
            wrapped_output_dim = self.wrapped_cell.state_size

        if self.return_attention:
            return input_shape[0], wrapped_output_dim + self.units
        else:
            return input_shape[0], wrapped_output_dim

    def _get_attended_shape(self):
        # TODO duplicates code in Layer...
        if isinstance(self.attended, list):
            attendeds = self.attended
            return_list = True
        else:
            attendeds = [self.attended]
            return_list = False

        attended_shapes = []
        for attended_element in attendeds:
            if hasattr(attended_element, '_keras_shape'):
                attended_shapes.append(attended_element._keras_shape)
            elif hasattr(K, 'int_shape'):
                attended_shapes.append(K.int_shape(attended_element))
            else:
                raise ValueError('You tried to call layer "' + self.name +
                                 '". This layer has no information'
                                 ' about its expected input shape, '
                                 'and thus cannot be built. '
                                 'You can build it manually via: '
                                 '`layer.build(batch_input_shape)`')
        if return_list:
            return attended_shapes
        else:
            # must be only one
            return attended_shapes[0]


def to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def get_shape(inputs):
    # TODO duplicates code in Layer...
    if isinstance(inputs, list):
        xs = inputs
        return_list = True
    else:
        xs = [inputs]
        return_list = False

    inputs_shape = []
    for x in xs:
        if hasattr(x, '_keras_shape'):
            inputs_shape.append(x._keras_shape)
        elif hasattr(K, 'int_shape'):
            inputs_shape.append(K.int_shape(x))
        else:
            raise ValueError('cannot infer shape of {}'.format(x))
    if return_list:
        return inputs_shape
    else:
        # must be only one
        return inputs_shape[0]
