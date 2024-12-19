'''
Author : Yan Shi (NPU Auto)
Date : 2023 / 7 / 15

MGBRNN 内部模型，在LSTM源码的基础上改的，从752行开始是改后的模型，前面的不用看
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import hashlib
import numbers
import tensorflow as tf
from tensorflow import reshape,transpose,cast,float64,float32,float16
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
_LSTMStateTuple_KF = collections.namedtuple("LSTMStateTuple", ("c", "h","P"))
sT = 1.

Tao = tf.cast(tf.stack([[[0.5*sT**2,0],[0,0.5*sT**2],[sT,0],[0,sT]],
              [[0.5*sT**2,0],[0,0.5*sT**2],[sT,0],[0,sT]],
              [[0.5*sT**2,0],[0,0.5*sT**2],[sT,0],[0,sT]],
              [[0.5*sT**2,0],[0,0.5*sT**2],[sT,0],[0,sT]],
              [[0.5*sT**2,0],[0,0.5*sT**2],[sT,0],[0,sT]],
              [[0.5*sT**2,0],[0,0.5*sT**2],[sT,0],[0,sT]]]),dtype=tf.float64)
def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.

  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).

  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.

  Returns:
    shape: the concatenation of prefix and suffix.

  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape

def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)



class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

class LSTMStateTuple_KF(_LSTMStateTuple_KF):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h,P) = self
    if c.dtype != h.dtype or c.dtype != P.dtype or P.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                      (str(c.dtype), str(h.dtype),str(P.dtype)))
    return c.dtype

class RNNCell(base_layer.Layer):
  """Abstract object representing an RNN cell.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.

  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    super(RNNCell, self).__init__(
        trainable=trainable, name=name, dtype=dtype, **kwargs)
    # Attribute that indicates whether the cell is a TF RNN cell, due the slight
    # difference between TF and Keras RNN cell.
    self._is_tf_rnn_cell = True

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(vs.get_variable_scope(),
                                  custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.executing_eagerly():
      trainable = variable._trainable  # pylint: disable=protected-access
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (isinstance(variable, tf_variables.PartitionedVariable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
      return (LSTMStateTuple(self._num_units,self._dim_state))

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      # Validate the given batch_size and dtype against inputs if provided.
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      if batch_size is not None:
        if tensor_util.is_tensor(batch_size):
          static_batch_size = tensor_util.constant_value(
              batch_size, partial=True)
        else:
          static_batch_size = batch_size
        if inputs.shape[0].value != static_batch_size:
          raise ValueError(
              "batch size from input tensor is different from the "
              "input param. Input tensor batch: {}, batch_size: {}".format(
                  inputs.shape[0].value, batch_size))

      if dtype is not None and inputs.dtype != dtype:
        raise ValueError(
            "dtype from input tensor is different from the "
            "input param. Input tensor dtype: {}, dtype: {}".format(
                inputs.dtype, dtype))

      batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
      dtype = inputs.dtype
    if None in [batch_size, dtype]:
      raise ValueError(
          "batch_size and dtype cannot be None while constructing initial "
          "state: batch_size={}, dtype={}".format(batch_size, dtype))
    return self.zero_state(batch_size, dtype)


  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
    # Try to use the last cached zero_state. This is done to avoid recreating
    # zeros, especially when eager execution is enabled.
    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and
          last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)


    return output

class LayerRNNCell(RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

  def __call__(self, inputs, state, scope=None, *args, **kwargs):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: optional cell scope.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(self, inputs, state, scope=scope,
                                     *args, **kwargs)


@tf_export(v1=["nn.rnn_cell.BasicRNNCell"])
class BasicRNNCell(LayerRNNCell):
  """The most basic RNN cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.

  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`. It could also be string
      that is within Keras activation function names.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

  @deprecated(None, "This class is equivalent as tf.keras.layers.SimpleRNNCell,"
                    " and will be replaced by that in Tensorflow 2.0.")
  def __init__(self,
               num_units,
               dim_state,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    super(BasicRNNCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BasicLSTMCell(LayerRNNCell):
  """DEPRECATED: Please use `tf.nn.rnn_cell.LSTMCell` instead.

  Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full `tf.nn.rnn_cell.LSTMCell`
  that follows.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

  @deprecated(None, "This class is deprecated, please use "
                    "tf.nn.rnn_cell.LSTMCell, which supports all the feature "
                    "this cell currently has. Please replace the existing code "
                    "with tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell').")
  def __init__(self,
               num_units,
               dim_state,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().

      When restoring from CudnnLSTM-trained checkpoints, must use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(BasicLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._dim_state = dim_state
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units,self._dim_state))

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME,shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(_BIAS_VARIABLE_NAME,shape=[4 * self._num_units],initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * num_units]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
    # 输入门
    gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel) # 乘以权重
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias) # 添加偏置

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    # 记忆更新门 （包含了遗忘门）
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),multiply(sigmoid(i), self._activation(j)))
    # 输出门
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "dim_state" : self._dim_state,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class LSTMCell(LayerRNNCell):
  """
  We implement EGBRNN using the original LSTM class.
  """

  def __init__(self, num_units,dim_state,dim_meas,
               trans_model,meas_model,meas_noise,pro_noise,
               sT,batch_size,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None, dtype=None, **kwargs):
    '''
    :param num_units: number of hidden node
    :param dim_state: state dimension
    :param dim_meas: measurement dimension
    :param trans_model: state transition function
    :param meas_model: state-measurement function
    :param meas_func: state-measurement function
    :param meas_matrix: state-measurement matrix
    :param meas_noise: measurement noise covariance
    :param pro_noise: process noise covariance
    :param sT: sampling interval
    :param batch_size: batch size
    '''
    super(LSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._dim_state = dim_state
    self._dim_meas = dim_meas
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._trans_model = trans_model
    self._meas_model = meas_model
    self._meas_noise = meas_noise
    self._pro_noise = pro_noise
    self._sT = sT
    self._batch_size = batch_size
    #
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._state_size = (LSTMStateTuple_KF(num_units,dim_state,dim_state*dim_state))
    self._output_size = num_units
  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    state_depth = self._dim_state
    meas_depth = self._dim_meas
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    '''Define the weights and biases of the network'''
    self._kernel_mem = self.add_variable("kernel_mem",shape=[state_depth + h_depth, self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._kernel_mod_l1 = self.add_variable("kernel_mod_l1",shape=[state_depth + h_depth, self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._kernel_mod_l2 = self.add_variable("kernel_mod_l2",shape=[self._num_units, 2],
        initializer=self._initializer,
        partitioner=maybe_partitioner)

    self._kernel_z_l1 = self.add_variable("kernel_z_l1",shape=[state_depth + h_depth+meas_depth, self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._kernel_z_l2 = self.add_variable("kernel_z_l2",shape=[self._num_units, 2],
        initializer=self._initializer,
        partitioner=maybe_partitioner)

    self._kernel_q_l1 = self.add_variable("kernel_q_l1",shape=[state_depth + h_depth, self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._kernel_q_l2 = self.add_variable("kernel_q_l2",shape=[self._num_units, 4],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._kernel_r_l1 = self.add_variable("kernel_r_l1",shape=[state_depth + h_depth + meas_depth, self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    self._kernel_r_l2 = self.add_variable("kernel_r_l2",shape=[self._num_units, 2],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    # self._kernel_memup = self.add_variable("kernel_memup",shape=[state_depth + h_depth + meas_depth, self._num_units],
    #     initializer=self._initializer,
    #     partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias_mem = self.add_variable("bias_mem",shape=[self._num_units],initializer=initializer)
    self._bias_mod_l1 = self.add_variable("bias_mod_l1", shape=[self._num_units], initializer=initializer)
    self._bias_mod_l2 = self.add_variable("bias_mod_l2", shape=[2], initializer=initializer)
    self._bias_z_l1 = self.add_variable("bias_z_l1", shape=[self._num_units], initializer=initializer)
    self._bias_z_l2 = self.add_variable("bias_z_l2", shape=[2], initializer=initializer)
    self._bias_q_l1 = self.add_variable("bias_q_l1", shape=[self._num_units], initializer=initializer)
    self._bias_q_l2 = self.add_variable("bias_q_l2", shape=[4], initializer=initializer)


    self._bias_r_l1 = self.add_variable("bias_r_l1", shape=[self._num_units], initializer=initializer)
    self._bias_r_l2 = self.add_variable("bias_r_l2", shape=[2], initializer=initializer)
    # self._bias_memup = self.add_variable("bias_memup",shape=[self._num_units],initializer=initializer)
    self.built = True

  def call(self, inputs, state):
    """
    Run one step of EGBRNN.
    Args:
      inputs: measurement [bt, meas_dim]
      state: A turple of (previous memory, previous state,previous covariance)
    Returns:
      - m : estimated state
      - new_state : A turplr of (updated memory, estimated state, estimated covariance)
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh
    (c_prev, m_prev,P_prev) = state
    P_prev = reshape(P_prev,[self._batch_size,self._dim_state,self._dim_state])
    '''MUG'''
    # Memory Update & Measure Network
    lstm_matrix_m = math_ops.matmul(array_ops.concat([c_prev, m_prev/60], 1), self._kernel_mem)
    c_update = sigmoid(nn_ops.bias_add(lstm_matrix_m, self._bias_mem))
    '''SPG'''
    # State Evolution Compensation Network
    lstm_matrix_mod_l1 = math_ops.matmul(array_ops.concat([c_update, m_prev/60], 1), self._kernel_mod_l1)
    lstm_matrix_mod_l1 = sigmoid(nn_ops.bias_add(lstm_matrix_mod_l1, self._bias_mod_l1))
    lstm_matrix_mod_l2 = math_ops.matmul(lstm_matrix_mod_l1, self._kernel_mod_l2)
    add = reshape(nn_ops.bias_add(lstm_matrix_mod_l2, self._bias_mod_l2),[self._batch_size,2,1])
    # State mean prediction
    m_pred = math_ops.matmul(self._trans_model,reshape(m_prev,[self._batch_size,self._dim_state,1])) + tf.matmul(Tao,add)
    # State Evolution Measure Network
    lstm_q_l1 = math_ops.matmul(array_ops.concat([c_update, m_prev/60], 1), self._kernel_q_l1)
    lstm_q_l1 = sigmoid(nn_ops.bias_add(lstm_q_l1, self._bias_q_l1))
    lstm_q_l2 = math_ops.matmul(lstm_q_l1, self._kernel_q_l2)
    q_est = reshape(nn_ops.bias_add(lstm_q_l2, self._bias_q_l2),[self._batch_size,1,self._dim_state])
    q_est_T = reshape(q_est,[self._batch_size,self._dim_state,1])
    Q_est = math_ops.matmul(q_est_T,q_est)
    # State covariance prediction
    P_pred = math_ops.matmul(math_ops.matmul(self._trans_model,P_prev),transpose(self._trans_model,[0,2,1]))+ self._pro_noise+ Q_est
    '''SUG'''
    # Measurement compensation network
    lstm_matrix_z_l1 = math_ops.matmul(array_ops.concat([c_update, m_prev/60,inputs/60], 1), self._kernel_z_l1)
    lstm_matrix_z_l1 = sigmoid(nn_ops.bias_add(lstm_matrix_z_l1, self._bias_z_l1))
    lstm_matrix_z_l2 = math_ops.matmul(lstm_matrix_z_l1, self._kernel_z_l2)
    z_add = reshape(nn_ops.bias_add(lstm_matrix_z_l2, self._bias_z_l2),[self._batch_size,2,1])
    # Residual
    v = reshape(inputs,[self._batch_size,self._dim_meas,1]) - (math_ops.matmul(self._meas_model,m_pred)+z_add)
    # Measuring Measure Network
    lstm_r_l1 = math_ops.matmul(array_ops.concat([c_update, m_prev/60,inputs/60], 1), self._kernel_r_l1)
    lstm_r_l1 = sigmoid(nn_ops.bias_add(lstm_r_l1, self._bias_r_l1))
    lstm_r_l2 = math_ops.matmul(lstm_r_l1, self._kernel_r_l2)
    r_est = reshape(nn_ops.bias_add(lstm_r_l2, self._bias_r_l2),[self._batch_size,1,self._dim_meas])
    r_est_T = reshape(r_est,[self._batch_size,self._dim_meas,1])
    R_est = math_ops.matmul(r_est_T,r_est)
    # Update of state and covariance
    S = math_ops.matmul(math_ops.matmul(self._meas_model,P_pred),transpose(self._meas_model,[0,2,1])) + R_est+self._meas_noise
    K = math_ops.matmul(math_ops.matmul(P_pred,transpose(self._meas_model,[0,2,1])),tf.linalg.inv(S))
    m = m_pred + math_ops.matmul(K,v)
    m = reshape(m,[self._batch_size,self._dim_state])
    P = P_pred - math_ops.matmul(math_ops.matmul(K,S),transpose(K,[0,2,1]))
    P = reshape(P,[self._batch_size,self._dim_state*self._dim_state])
    c = c_update
    new_state = (LSTMStateTuple_KF(c, m,P) if self._state_is_tuple else
                 array_ops.concat([c, m,P], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))