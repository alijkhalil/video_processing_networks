# Import statements
import sys
import numpy as np
import tensorflow as tf

from keras import backend as K

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers

from keras.models import Model, Sequential
from keras.engine.topology import InputSpec, Layer
from keras.layers import Input, Dense, Lambda, Add, Flatten
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Recurrent, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.legacy import interfaces
from keras.layers.core import Activation

sys.path.append("..")
from dl_utilities.layers import general as dl_layers
from dl_utilities.layers import pcn as pcn_layers

 
    
# Global variables
GLOBAL_AVERAGE_POOL_STR='global'
                        
MODEL_STEM_INDEX = 0
MODEL_NUM_STEM_CONV = 3

MODEL_TIME_STEPS_P_BLOCK=np.array([1, 1, 1, 1, 3, 1, 6, 1])
MODEL_STATES_P_BLOCK = 2
MODEL_NUM_BLOCKS = 8

assert len(MODEL_TIME_STEPS_P_BLOCK) == MODEL_NUM_BLOCKS
MODEL_TOTAL_TIME_STEPS=np.sum(MODEL_TIME_STEPS_P_BLOCK)
MODEL_TOTAL_STATES=(MODEL_TOTAL_TIME_STEPS * MODEL_STATES_P_BLOCK)
             
DEFAULT_WEIGHT_DECAY=1E-4



# Simple helper to get current state index based on block index
def get_starting_state_index(block_index):
    num_states = int(np.sum(MODEL_TIME_STEPS_P_BLOCK[:block_index]))
    return num_states * MODEL_STATES_P_BLOCK
            
            
    
# Predictive Corrective Network definition
class PCN_Cell(Recurrent):
    """
    The PCN cell represents the core logic for the Predictive Corrective Network 
    model.  At a high level, the model can be expressed as an RNN with a receptive 
    field across 6 time steps.  It is based on the underlying theory of linear 
    dynamic systems for modeling time series involved in Kalman Filters.  
    Ultimately, the model aims to predict actions in a sequence of individual frames 
    in a video.  However, the model can be refactored to either provide a video-level 
    label or provide a video-level embedding by fine-tuning the PCN parameters. 
    
    # Issues:
        -Instability of training RNN's on Keras
            -particularly with high dropout rate
        -size of model and inability to train larger batches/time-series
           -requires extensive computing powers (e.g. many high-end GPUs)
        
    # Arguments
        output_units: Positive integer
            -dimensionality of the output state space
        block_filters: List of 9 integers
            -output filters for convolutions in each of the 8 "blocks" (plus the initial stem)
            -first value must be at least 32
        downsize_block_indices: List of non-negative integers (no larger than 7) 
            -represents indices of "blocks" requiring an initial 2x2 resolution downsize
        final_downsize_approach: Option from the elements below
            -None: no downsizing after 8th PCN block
            -'global': apply a global pool to reduce spatial dimensions to 1x1
            -tuple: an average pooling with that pool size (and the same stride size)
        num_conv_p_kalman_filter: Positive integer
            -number of separable convolution operations for each Kalman Filter function
        num_res_connect_p_block: Positive integer
            -number of residual blocks for post-processing after each "block" in model
        num_final_dense_layers: Positive integer
            -number of dense layers to conclude the model
        frame_level_output: a Boolean value
            -flag to return each frame's output or only final output embedding at last time step
            
    # References
        - [PCN] (https://arxiv.org/pdf/1704.03615.pdf)
    """
    
    @interfaces.legacy_recurrent_support
    def __init__(self, 
                 output_units, 
                 block_filters=[32, 64, 64, 128, 128, 256, 256, 512, 512],
                 downsize_block_indices=[0, 2, 6],
                 final_downsize_approach=GLOBAL_AVERAGE_POOL_STR,
                 num_conv_p_kalman_filter=2,
                 num_res_connect_p_block=4,
                 num_final_dense_layers=3,
                 final_dropout=0.1,
                 frame_level_output=False,
                 **kwargs):
         
        super(PCN_Cell, self).__init__(**kwargs)
        
        # Perform checks on inputs
        if output_units is None or output_units < 0:
            raise ValueError("The 'output_units' variable must be a positive integer.")
                
        if type(block_filters) is not list:      
            raise ValueError("The 'block_filters' variable must be a list.")    

        if len(block_filters) != 9:
            raise ValueError("The 'block_filters' list should contain exactly 9 items.")
            
        prev_el = 16        
        for el in block_filters:
            if prev_el <= el:
                prev_el = el
            else:    
                raise ValueError("The 'block_filters' element values must never "
                                    "decrease and start at a minimum of 16.")  
            
        if type(downsize_block_indices) is not list:        
            raise ValueError("The 'downsize_blocks' variable must be a list.")
        
        if len(np.unique(downsize_block_indices)) != len(downsize_block_indices):
            raise ValueError("The 'downsize_blocks' list should not have duplicate values.")

        if num_final_dense_layers < 2:
            raise ValueError("The 'num_final_dense_layers' value should be at least 2.")
            
        for el in downsize_block_indices:
            if el > 7 or el < 0:
                raise ValueError("The 'downsize_blocks' indicies should be a maximum value of 7.")

        if (final_downsize_approach is not None and 
                final_downsize_approach != GLOBAL_AVERAGE_POOL_STR and  
                final_downsize_approach is not tuple):
            raise ValueError("The 'final_downsize_approach' variable must be either be None, "
                                "a tuple, or ")   
            
                                
        # Assign member variables
        self.internal_layers = {}
        self.input_spec[0].ndim = 5
        
        self.output_units = output_units
        self.block_filters = block_filters
        
        self.downsize_block_indices = sorted(downsize_block_indices, key=int)
        self.final_downsize_approach = final_downsize_approach                
        
        self.num_conv_p_kalman_filter = num_conv_p_kalman_filter
        self.num_res_connect_p_block = num_res_connect_p_block
        self.num_final_dense_layers = num_final_dense_layers
        
        self.dropout = min(0.9, max(0., final_dropout))
        self.return_sequences = frame_level_output
        

    # Override to ensure that implementation does rely on Keras to have 'return_state' support
    def call(self, inputs, mask=None, initial_state=None, training=None):
        if initial_state is not None:
            if not isinstance(initial_state, (list, tuple)):
                initial_states = [initial_state]
            else:
                initial_states = list(initial_state)
                
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_state(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
                             
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
                             
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
                                             
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if not isinstance(states, (list, tuple)):
            states = [states]
        else:
            states = list(states)
            
        # Return output and states combined (in a list)    
        return [output] + states

    
    # No suppport for "add_weight" - instead use higher-level Layers and "add_layer" function
    def add_weight(self, shape, initializer,
                   name=None,
                   trainable=True,
                   regularizer=None,
                   constraint=None):
                   
        raise ValueError("The 'add_weight' function is not allowed for this " \
                            "particular RNN layer. Use 'add_layer' function instead.")                 
    
    
    # Layer functions
    def add_layer(self, layer, name, input_shape):
        self.internal_layers[name] = layer
        self.internal_layers[name].build(input_shape)
        
        
    def get_layer(self, name):
        return self.internal_layers[name]
    
    
    # Weight override function 
    @property
    def trainable_weights(self):
        tmp_weights = []
        
        for i_layer in self.internal_layers.values():
            tmp_weights.extend(i_layer.trainable_weights)
    
        return tmp_weights

    @property
    def non_trainable_weights(self):
        tmp_weights = []
        
        for i_layer in self.internal_layers.values():
            tmp_weights.extend(i_layer.non_trainable_weights)
    
        return tmp_weights
        
    
    # Override generic RNN functions because states are unique for this cell
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Output shape
        batch_size = input_shape[0]    
        if self.return_sequences:
            output_shape = (batch_size, input_shape[1], self.output_units)
        else:
            output_shape = (batch_size, self.output_units)

        # State shapes            
        state_shape = [ ]
        for i_spec in self.state_spec:
            cur_dim = i_spec.shape
            state_shape.append((batch_size, ) + cur_dim[1:])
            
        # Return concatenation of the two shapes    
        return [output_shape] + state_shape
    
            
    def get_initial_state(self, inputs):
        # Build an all-zero tensor of shape (samples, 1)          
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, ) + img_dims
        initial_state = K.sum(initial_state, axis=(1, 2, 3, 4), keepdims=True)  # (samples, 1, 1, 1, 1)
        initial_state = K.squeeze(initial_state, axis=-1)  # (samples, 1, 1, 1)
        
        # Build zero-ed intermediate states by getting dimension of each state
        initial_states = []
        for i_spec in self.state_spec:
            cur_dim = i_spec.shape
            tile_shape = list((1, ) + cur_dim[1:])
            
            tmp_state = K.tile(initial_state, tile_shape)  # (samples, ) + state_dim
            initial_states.append(tmp_state)
        
        # Return them
        return initial_states
    
    
    def reset_states(self, states_value=None):
        if not self.stateful:
            raise AttributeError('PCN_Cell must be stateful.')
            
        if not self.input_spec:
            raise RuntimeError('PCN_Cell has never been called '
                                'and thus has no states.')
                               
        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
                             
        if states_value is not None:
            if not isinstance(states_value, (list, tuple)):
                states_value = [states_value]
                
            if len(states_value) != len(self.states):
                raise ValueError('The layer has ' + str(len(self.states)) +
                                 ' states, but the `states_value` '
                                 'argument passed '
                                 'only has ' + str(len(states_value)) +
                                 ' entries')
                                 
        if self.states[0] is None:
            self.states = []
            for i_spec in self.state_spec:
                cur_dim = i_spec.shape
                self.states.append(K.zeros((batch_size, ) + cur_dim[1:]))
                
            if not states_value:
                return
                
        for i, state_tuple in enumerate(zip(self.states, self.state_spec)):
            state, tmp_state_spec = state_tuple
            cur_dim = tmp_state_spec.shape
            
            tmp_state_shape = (batch_size, ) + cur_dim[1:]
            if states_value:
                value = states_value[i]

                if value.shape != tmp_state_shape:
                    raise ValueError(
                        'Expected state #' + str(i) +
                        ' to have shape ' + str(tmp_state_shape) +
                        ' but got array with shape ' + str(value.shape))
            else:
                value = np.zeros(tmp_state_shape)
                
            K.set_value(state, value)
    
    
    # Should contain all layers with weights associated with them
    def build(self, input_shape):
        # Get relavent dimension values
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        batch_size = input_shape[0] if self.stateful else None
        orig_img_dim = input_shape[2:]
        
        
        # Set state shapes
        init_shape = (batch_size, ) + orig_img_dim
        state_shapes = [ ]
        
        for i, num_channels in enumerate(self.block_filters[:-1]):
            start_index = get_starting_state_index(i)
            end_index = get_starting_state_index(i+1)
            
            num_iters = end_index - start_index
            for j in range(num_iters):
                state_shapes.append(init_shape[:-1] + (num_channels, ))
        
        
        for downsize_i in self.downsize_block_indices:
            cur_i = get_starting_state_index(downsize_i)
            
            orig_shape = state_shapes[cur_i]
            h_val = int((orig_shape[1] + 1) // 2)
            w_val = int((orig_shape[2] + 1) // 2)
        
            for i in range(cur_i, MODEL_TOTAL_STATES):
                num_channels = state_shapes[i][-1]
                state_shapes[i] = (batch_size, h_val, w_val, num_channels)
                
                
        # Set input and state spec values 
        self.input_spec = InputSpec(shape=((batch_size, None) + orig_img_dim))         
        self.state_spec = [ InputSpec(shape=cur_shape) for cur_shape in state_shapes ]    # states

        
        # Initialize states to None
        self.states = [ None ] * MODEL_TOTAL_STATES
        if self.stateful:
            self.reset_states()

            
        # Define stem layers (e.g. simple 2D conv)
        updated_shape = init_shape[:-1] + (self.block_filters[MODEL_STEM_INDEX], )
        num_channels = updated_shape[-1]
        
        for i in range(MODEL_NUM_STEM_CONV):
            # Add conv layers here
            weight_name = ('stem_conv_kernel_%d' % i) 
            layer_id = ('stem_conv_layer_%d' % i)
            
            tmp_layer = Conv2D(num_channels, (3, 3), kernel_initializer='he_uniform', 
                                    padding='same', use_bias=False, 
                                    kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                    name=weight_name)
                                    
            if i == 0:
                self.add_layer(tmp_layer, layer_id, init_shape)
            else:
                self.add_layer(tmp_layer, layer_id, updated_shape)
                
            # Add Batch Norm layer here
            weight_name = ('stem_bn_kernel_%d' % i) 
            layer_id = ('stem_bn_layer_%d' % i)
            
            tmp_layer = BatchNormalization(name=weight_name)
            
            self.add_layer(tmp_layer, layer_id, updated_shape)                                                                 
        
        
        # Add layers for remaining blocks
        next_channels = 0
        
        for i in range(MODEL_NUM_BLOCKS):
            # Get number of channels for the block
            cur_shape = state_shapes[get_starting_state_index(i)]
            num_channels = cur_shape[-1]
            
            is_last = False
            if (i+1) == MODEL_NUM_BLOCKS:
                is_last = True
            else:
                next_state_i = get_starting_state_index(i+1)
                next_channels = state_shapes[next_state_i][-1]
            
            
            # Tanh layer function used to modulate activations between time steps
            weight_name = ('tanh_weights_%d' % i) 
            layer_id = ('tanh_layer_%d' % i) 

            tmp_layer = pcn_layers.GetTanhValue(name=weight_name)
            self.add_layer(tmp_layer, layer_id, cur_shape)            
            
            
            # Conv weights operating on the difference between time steps
            for j in range(self.num_conv_p_kalman_filter):
                # Define conv layers here
                weight_name = ('kalman_conv_kernel_%d_%d' % (i, j)) 
                layer_id = ('kalman_conv_layer_%d_%d' % (i, j))
            
                tmp_layer = SeparableConv2D(num_channels, (3, 3), kernel_initializer='he_uniform', 
                                                padding='same', use_bias=False, 
                                                kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                                name=weight_name)
                                        
                self.add_layer(tmp_layer, layer_id, cur_shape)
        
                # Define Batch Norm layers
                weight_name = ('kalman_bn_kernel_%d_%d' % (i, j)) 
                layer_id = ('kalman_bn_layer_%d_%d' % (i, j))
                
                tmp_layer = BatchNormalization(name=weight_name)
                
                self.add_layer(tmp_layer, layer_id, cur_shape)

                
            # Post processing residual connections (after addition)
            for j in range(self.num_res_connect_p_block):
                downsize_channels = int(num_channels // 2)
                downsize_shape = cur_shape[:-1] + (downsize_channels, )
                
                # Add residual convolutions here
                weight_name = ('res_conv_kernel_%d_%d_1' % (i, j)) 
                layer_id = ('res_conv_layer_%d_%d_1' % (i, j))
                
                tmp_layer = SeparableConv2D(downsize_channels, (3, 3), kernel_initializer='he_uniform', 
                                                            padding='same', use_bias=False, 
                                                            kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                                            name=weight_name)
                                        
                self.add_layer(tmp_layer, layer_id, cur_shape)
                
                weight_name = ('res_conv_kernel_%d_%d_2' % (i, j)) 
                layer_id = ('res_conv_layer_%d_%d_2' % (i, j))
                
                tmp_layer = SeparableConv2D(num_channels, (3, 3), kernel_initializer='he_uniform', 
                                                            padding='same', use_bias=False, 
                                                            kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                                            name=weight_name)
                                        
                self.add_layer(tmp_layer, layer_id, downsize_shape)
            
                # Add residual Batch Norm here
                weight_name = ('res_bn_kernel_%d_%d_1' % (i, j)) 
                layer_id = ('res_bn_layer_%d_%d_1' % (i, j))
                
                tmp_layer = BatchNormalization(name=weight_name)
                
                self.add_layer(tmp_layer, layer_id, downsize_shape)                                    

                weight_name = ('res_bn_kernel_%d_%d_2' % (i, j)) 
                layer_id = ('res_bn_layer_%d_%d_2' % (i, j))
                
                tmp_layer = BatchNormalization(name=weight_name)
                
                self.add_layer(tmp_layer, layer_id, cur_shape)

                
            # Final convolution to upscale number of filters if needed
            if not is_last:
                weight_name = ('final_conv_kernel_%d' % i) 
                layer_id = ('final_conv_layer_%d' % i)
                
                tmp_layer = SeparableConv2D(next_channels, (3, 3), kernel_initializer='he_uniform', 
                                                padding='same', use_bias=False, 
                                                kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                                name=weight_name)
                                        
                self.add_layer(tmp_layer, layer_id, cur_shape)
            
            
        # Add final 'num_final_dense_layers' Dense layers
        final_shape = state_shapes[-1][1:]
        if self.final_downsize_approach is not None:
            if self.final_downsize_approach == GLOBAL_AVERAGE_POOL_STR:
                final_shape = (1, 1, final_shape[2])
            else:
                h_val = final_shape[0] // self.final_downsize_approach[0]
                w_val = final_shape[1] // self.final_downsize_approach[1]
                final_shape = (h_val, w_val, final_shape[2])
        
        final_channels = 1
        for dim in final_shape:
            final_channels *= dim
            
        flattened_shape = (batch_size, final_channels)
        
        diff = int((self.output_units - final_channels) // 2)
        intermediate_channels = final_channels + diff
        self.intermediate_shape = (batch_size, intermediate_channels)
        
        
        for i in range(self.num_final_dense_layers - 1):
            weight_name = ('final_dense_kernel_%d' % i) 
            layer_id = ('final_dense_layer_%d' % i)
            
            tmp_layer = Dense(intermediate_channels, use_bias=False, 
                                            kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY), 
                                            name=weight_name)
            if i == 0:                         
                self.add_layer(tmp_layer, layer_id, flattened_shape)
            else:
                self.add_layer(tmp_layer, layer_id, self.intermediate_shape)
                
        weight_name = ('final_dense_kernel_%d' % (self.num_final_dense_layers - 1)) 
        layer_id = ('final_dense_layer_%d' % (self.num_final_dense_layers - 1))
        
        tmp_layer = Dense(self.output_units, use_bias=False, 
                                    kernel_regularizer=l2(DEFAULT_WEIGHT_DECAY),
                                    name=weight_name)
                                
        self.add_layer(tmp_layer, layer_id, self.intermediate_shape)
        
        
        # Set built flag
        self.built = True
        

    # Called immediately before RNN step as part of set-up process
    # Passed as "state" element (after output and intermediary states) as result of RNN iteration
    # Normally used to pass dropout masks
    def get_constants(self, inputs, training=None):
        constants = []
        tile_shape = list((1, ) + self.intermediate_shape[1:])
        
        # Set ones tensor with shape of hidden layer
        ones = K.ones_like(K.reshape(inputs[:, 0, 0, 0, 0], (-1, 1)))   # (samples, 1)
        for _ in range(2, len(tile_shape)):    # (samples, 1, ...., 1)
            ones = K.expand_dims(ones)
            
        ones = K.tile(ones, tile_shape)  # Now it is the same shape as the second to last layer
        
        # Get input and recurrent dropout masks
        if 0.0 < self.dropout < 1.0:            
            dp_mask = K.in_train_phase(K.dropout(ones, self.dropout),
                                        ones,
                                        training=training)                                                
                                            
        else:
            dp_mask = ones
        
        constants.append(dp_mask)
        
        
        # Return them        
        return constants

        
    def step(self, inputs, states):
        # Break down previous output/states
        core_states = states[:-1]
        dp_mask = states[-1]		# from "get_constants"

        
        # Pass images through stem (with optional downsize)
        new_states = []
        
        stem = inputs
        for i in range(MODEL_NUM_STEM_CONV):
            conv_layer_id = ('stem_conv_layer_%d' % i)
            bn_layer_id = ('stem_bn_layer_%d' % i)
            
            stem = self.get_layer(conv_layer_id)(stem)
            stem = self.get_layer(bn_layer_id)(stem)
            
            if (i+1) == MODEL_NUM_STEM_CONV:
                if MODEL_STEM_INDEX in self.downsize_block_indices:
                    stem = AveragePooling2D()(stem)
            else:
                stem = Activation('relu')(stem)        
                
        new_states.append(stem)
        
        
        # Pass through PCN "blocks"    
        x = stem
        for i in range(MODEL_NUM_BLOCKS):
            # Get subtraction value
            start_index = get_starting_state_index(i)

            norm_stem = pcn_layers.NormalizePerChannel()(x)
            norm_prev_stem = pcn_layers.NormalizePerChannel()(core_states[start_index])
            x = dl_layers.Subtract()([norm_stem, norm_prev_stem])

            
            # Get tanh-based scalar value (for each example in batch)
            tanh_layer_id = ('tanh_layer_%d' % i) 
            tahn_output = self.get_layer(tanh_layer_id)(x)
            
            
            # Pass substract layer through the Kalman filter MLP
            for j in range(self.num_conv_p_kalman_filter):
                conv_layer_id = ('kalman_conv_layer_%d_%d' % (i, j))
                bn_layer_id = ('kalman_bn_layer_%d_%d' % (i, j))
                
                x = self.get_layer(conv_layer_id)(x)
                x = self.get_layer(bn_layer_id)(x)
                if (j+1) != self.num_conv_p_kalman_filter:
                    x = Activation('relu')(x)
                        
            new_states.append(x)
            
            
            # Get addition combinition of two time steps (based on tanh value)
            x = pcn_layers.GetTanhCombination()([tahn_output, x, core_states[start_index+1]])
            
            
            # Post processing residual connections (after addition)
            for j in range(self.num_res_connect_p_block):                
                conv_layer_id_1 = ('res_conv_layer_%d_%d_1' % (i, j))
                conv_layer_id_2 = ('res_conv_layer_%d_%d_2' % (i, j))
                bn_layer_id_1 = ('res_bn_layer_%d_%d_1' % (i, j))
                bn_layer_id_2 = ('res_bn_layer_%d_%d_2' % (i, j))
                
                res_term = Activation('relu')(x)
                res_term = self.get_layer(conv_layer_id_1)(res_term)
                res_term = self.get_layer(bn_layer_id_1)(res_term)

                res_term = Activation('relu')(res_term)
                res_term = self.get_layer(conv_layer_id_2)(res_term)
                res_term = self.get_layer(bn_layer_id_2)(res_term)
            
                x = Add()([x, res_term])
            
            
            # Final convolution and to upscale number of filters if needed
            if (i+1) != MODEL_NUM_BLOCKS:
                conv_layer_id = ('final_conv_layer_%d' % i)
                x = self.get_layer(conv_layer_id)(x)
                                
                if (i+1) in self.downsize_block_indices:
                    x = AveragePooling2D()(x)
            
                new_states.append(x)            
                            
            
        # Downsize final convolution layer if needed
        if self.final_downsize_approach is not None:
            if self.final_downsize_approach == GLOBAL_AVERAGE_POOL_STR:
                x = GlobalAveragePooling2D()(x)
            else:
                x = AveragePooling2D(self.final_downsize_approach)(x)
            
        
        # Apply final fully connected layers    
        if len(K.int_shape(x)) > 2:
            x = Flatten()(x)
    
        for i in range(self.num_final_dense_layers):
            final_layer_id = ('final_dense_layer_%d' % i)
            
            final_output = self.get_layer(final_layer_id)(x)
            
            if (i+1) != self.num_final_dense_layers:
                x = Activation('relu')(final_output)
                if 0.0 < self.dropout:
                    x = x * dp_mask
        
        # Re-organize old states and add new ones
        for i in range(MODEL_NUM_BLOCKS):
            start_i = get_starting_state_index(i)
            next_i = get_starting_state_index(i+1)
            
            cur_time_end_i = next_i - MODEL_STATES_P_BLOCK
            prev_time_start_i = start_i + MODEL_STATES_P_BLOCK
            
            for count, new_i in enumerate(range(start_i, cur_time_end_i)):
                new_states.insert(new_i, states[prev_time_start_i + count])
            
            
        # Set learning phase flag
        if 0.0 < self.dropout:
            final_output._uses_learning_phase = True
        
        
        # Return output and updated states
        return final_output, new_states

        
    def get_config(self):    
        config = {'output_units': self.output_units,
                  'block_filters': self.block_filters,
                  'downsize_block_indices': self.downsize_block_indices,
                  'final_downsize_approach': self.final_downsize_approach,
                  'num_conv_p_kalman_filter': self.num_conv_p_kalman_filter,
                  'num_res_connect_p_block': self.num_res_connect_p_block,
                  'num_final_dense_layers': self.num_final_dense_layers,
                  'dropout': self.dropout}
                  
        base_config = super(PCN_Cell, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))    
        
        
        
        
        
        
################ MAIN ################
if __name__ == '__main__':

    time_steps = 20
    img_dimensions = (32, 32, 3)
    num_labels = 100

    '''
    https://github.com/achalddave/predictive-corrective

    https://github.com/achalddave/predictive-corrective/blob/master/download.sh
    https://github.com/achalddave/thumos-scripts/blob/master/parse_temporal_annotations_to_hdf5.py

    http://www.thumos.info/download.html
    '''       
    
    input = Input((time_steps, ) + img_dimensions)

    pcn_cell = PCN_Cell(num_labels)
    out_layers = pcn_cell(input)

    final_layer = out_layers[0]
    final_preds = Activation('softmax')(final_layer)

    model = Model(input, final_preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()                         
    print("Done.")
