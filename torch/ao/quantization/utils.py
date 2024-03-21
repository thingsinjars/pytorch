"""
Utils shared by different modes of quantization (eager/graph)
"""
import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized

NodePattern = Union[Tuple[Node, Node], Tuple[Node, Tuple[Node, Node]], Any]
NodePattern.__module__ = "torch.ao.quantization.utils"

# This is the Quantizer class instance from torch/quantization/fx/quantize.py.
# Define separately to prevent circular imports.
# TODO(future PR): improve this.
# make this public once fixed (can't be public as is because setting the module directly
# doesn't work)
QuantizerCls = Any

# Type for fusion patterns, it can be more complicated than the following actually,
# see pattern.md for docs
# TODO: not sure if typing supports recursive data types
Pattern = Union[
    Callable, Tuple[Callable, Callable], Tuple[Callable, Tuple[Callable, Callable]], Any
]
Pattern.__module__ = "torch.ao.quantization.utils"

# TODO: maybe rename this to MatchInputNode
class MatchAllNode:
    """ A node pattern that matches all nodes, used in defining
    fusion patterns in FX Graph Mode Quantization
    """
    pass

module_type_list = {
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
    torch.nn.Identity,
    torch.nn.Hardsigmoid,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
}
func_list = {
    torch.nn.functional.adaptive_avg_pool1d,
    torch.nn.functional.adaptive_avg_pool2d,
    torch.nn.functional.adaptive_avg_pool3d,
    torch.nn.functional.elu,
    torch.nn.functional.hardswish,
    torch.nn.functional.instance_norm,
    torch.nn.functional.layer_norm,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.silu,
    torch.nn.functional.mish,
    torch.nn.functional.dropout,
    torch.nn.functional.max_pool1d,
    torch.nn.functional.max_pool2d,
    torch.nn.functional.max_pool3d,
    torch.nn.functional.relu,
    torch.nn.functional.hardtanh,
    torch.nn.functional.hardtanh_,
    torch.nn.functional.hardsigmoid,
    torch.nn.functional.sigmoid,
    torch.transpose,
    torch.repeat_interleave,
    torch.sigmoid,
    torch.squeeze,
    torch.stack,
    torch.sum,
    torch.tanh,
    torch.unsqueeze,
    torch.cat,
}
method_list = {
    torch.mean,
    'relu',
    'relu_',
    'contiguous',
    'detach',
    'detach_',
    'hardsigmoid',
    'hardsigmoid_',
    'permute',
    'repeat',
    'repeat_interleave',
    'reshape',
    'resize_',
    'shape',
    'sigmoid',
    'sigmoid_',
    'size',
    'squeeze',
    'squeeze_',
    'tanh',
    'tanh_',
    'transpose',
    'unsqueeze',
    'unsqueeze_',
    'view',
}

# TODO: not used now, remove
def check_node(node, modules):
    # TODO: reuse is_fixed_qparam_node after we move this function to _lower_to_native_backend.py
    """
    determines whether a given Python node can be optimized by calling another
    module, function, or method. It returns three booleans indicating which type
    of optimization is possible.

    Args:
        node (Node instance.): Python abstract syntax tree ( AST) node being checked
            for calls to functions, methods, or modules.
            
            		- `op`: The operation performed by the node, which is a string
            representing one of "call_function", "call_method", or "call_module".
            		- `target`: The target of the operation, which can be a function
            name, method name, or module name.
            		- `modules`: A dictionary containing the modules associated with the
            target, where each key is a string representing a module name, and
            each value is the type of the module (one of the types in the `module_type_list`).
            
        modules (list): list of modules that are relevant to the operation of the
            `check_node` function.

    Returns:
        bool: a list of boolean values indicating whether each operation on the
        given node can be performed.

    """
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    return is_call_function, is_call_method, is_call_module

def get_combined_dict(default_dict, additional_dict):
    """
    combines two dictionaries, creating a new dictionary that contains all the
    keys and values from both sources. It uses the `copy()` method to create a new
    instance of the `default_dict` and then updates it with the contents of the
    `additional_dict`. The resulting dictionary is returned in the function's
    return statement.

    Args:
        default_dict (dict): initial set of dict values that will be combined with
            the values from `additional_dict`.
        additional_dict (dict): additional dict to be merged with the default dict
            to produce the final combined dict returned by the `get_combined_dict`
            function.

    Returns:
        dict: a new dictionary that contains the contents of both the default and
        additional dictionaries after being updated.

    """
    d = default_dict.copy()
    d.update(additional_dict)
    return d

def is_per_tensor(qscheme):
    """
    determines whether a given tensor scheme is either affine or symmetric per-tensor,
    based on the value of the `qscheme` variable.

    Args:
        qscheme (`torch.Tensor` object.): 3D tensor transformation scheme, which
            is used to determine if the given 3D tensor can be transformed using
            an affine or symmetric operation.
            
            		- `torch.per_tensor_affine`: Indicates that the tensor is affine
            transformable across all dimensions.
            		- `torch.per_tensor_symmetric`: Indicates that the tensor is symmetric.
            
            	The function returns `True` if either of these properties is satisfied,
            and `False` otherwise.
            

    Returns:
        bool: a boolean indicating whether the given tensor scheme is per-tensor
        affine or symmetric.

    """
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def is_per_channel(qscheme):
    """
    determines if a given scheme is a per-channel affine transformation, specifically
    checking if it belongs to one of three predefined schemes: `torch.per_channel_affine`,
    `torch.per_channel_affine_float_qparams`, or `torch.per_channel_symmetric`.

    Args:
        qscheme (`torch.Schedule`.): 3-element list of affine parameters for
            per-channel transformations, which the function checks to determine
            if it belongs to one of the specified schemes.
            
            		- `torch.per_channel_affine`: This scheme enables per-channel affine
            transformations during inference. It allows for modifications to
            individual channels within a tensor, which can be useful in various applications.
            		- `torch.per_channel_affine_float_qparams`: This is an extended
            version of the `per_channel_affine` scheme that additionally provides
            float-point quantization parameters for each channel. This feature
            enables precision control during inference and can help reduce
            computational resources.
            		- `torch.per_channel_symmetric`: This scheme facilitates symmetric
            transformations across all channels within a tensor. It is useful in
            applications where symmetry is crucial, such as image processing tasks.
            

    Returns:
        bool: a boolean indicating whether the given scheme is a per-channel affine
        transformation.

    """
    return qscheme in [torch.per_channel_affine,
                       torch.per_channel_affine_float_qparams,
                       torch.per_channel_symmetric]

def getattr_from_fqn(obj: Any, fqn: str) -> Any:
    """
    Given an obj and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    return functools.reduce(getattr, fqn.split("."), obj)

def to_underlying_dtype(qdtype):
    """
    maps a quantum-related data type to its underlying classical data type, based
    on a mapping table.

    Args:
        qdtype (dict): 1-dimensional or higher tensor data type that is being
            converted to its underlying dtype.

    Returns:
        int: the underlying dtype of a given Quartus-defined datatype.

    """
    DTYPE_MAPPING = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
        torch.quint4x2: torch.uint8,
        torch.quint2x4: torch.uint8,
        torch.uint8: torch.uint8,
        torch.int8: torch.int8,
        torch.int16: torch.int16,
        torch.int32: torch.int32,
    }
    assert qdtype in DTYPE_MAPPING, "Unsupported dtype: " + str(qdtype)
    return DTYPE_MAPPING[qdtype]

def get_qparam_dict(observer_or_fake_quant):
    """
    generates a dictionary of quantization parameters for a given observer or fake
    quantizer object. It determines the qscheme based on the input, and then
    calculates the scale, zero point, and other quantization parameters using the
    observer's/fake quantizer's calculate_qparams method.

    Args:
        observer_or_fake_quant (observer or fake quantization object, which can
            be of any valid Python object type.): observer or fake quantizer that
            the function is working with, and it provides information about the
            quantization scheme and parameters for the corresponding tensor.
            
            		- `qscheme`: The quantization scheme applied to the tensor. It can
            be one of `"per_tensor"`, `"per_channel"` or `"affine"` indicating
            whether the quantization is performed per tensor, per channel or affine
            transformation.
            		- `dtype`: The data type of the tensor.
            		- `axis`: The axis along which the quantization is applied for
            per-channel quantization.
            		- `calculate_qparams()`: A method that returns the scale and zero
            point of the quantized tensor.
            		- `quant_min` and `quant_max`: The minimum and maximum values of the
            quantized tensor, respectively.
            

    Returns:
        dict: a dictionary containing parameters for quantization, including
        `qscheme`, `dtype`, `axis`, `scale`, `zero_point`, and additional parameters
        `quant_min` and `quant_max`.

    """
    from torch.ao.quantization.observer import PlaceholderObserver

    qscheme = getattr(observer_or_fake_quant, "qscheme", None)
    dtype = observer_or_fake_quant.dtype
    qparams = {"qscheme": qscheme, "dtype": dtype}

    if not qscheme or isinstance(observer_or_fake_quant, PlaceholderObserver):
        return {"qscheme": None, "dtype": dtype}

    if is_per_tensor(qscheme):
        qscheme = torch.per_tensor_affine
    elif is_per_channel(qscheme):
        # change symmetric to affine since we do not have symmetric
        # quantized Tensor
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine
        qparams["axis"] = observer_or_fake_quant.ch_axis
    else:
        raise RuntimeError(f"Unrecognized qscheme: {qscheme}")
    # update qscheme, since we don't have symmetric quant qscheme
    # in quantized Tensor
    qparams["qscheme"] = qscheme

    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams["scale"] = scale
    qparams["zero_point"] = zero_point

    if hasattr(observer_or_fake_quant, "quant_min"):
        qparams["quant_min"] = observer_or_fake_quant.quant_min
    if hasattr(observer_or_fake_quant, "quant_max"):
        qparams["quant_max"] = observer_or_fake_quant.quant_max

    return qparams


def get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """
    maps a custom module to its corresponding observed module class based on a
    mapping, and checks if the input custom module is of a supported type for the
    mapping.

    Args:
        custom_module (observed module class.): custom module for which the
            corresponding observed module class needs to be retrieved from the mapping.
            
            		- `quant_type`: The quantum type associated with the custom module.
            This is determined by the `qconfig` argument passed to the function.
            		- `class_mapping`: A mapping of quantum types to corresponding
            observed module classes. This mapping is retrieved from the
            `custom_module_class_mapping` argument. If the specified quantum type
            is not found in the mapping, a `KeyError` is raised.
            		- `type(custom_module)`: The Python type of the input `custom_module`.
            This is used to determine whether it corresponds to an observed module
            class in the `class_mapping`.
            
        custom_module_class_mapping (dict): mapping between observed module types
            and their corresponding custom classes, as determined by the `qconfig`.
        qconfig (object of quantitative configuration.): configuration for
            quantization, which is used to determine the appropriate observed
            module class for custom modules based on their type.
            
            		- `quant_type`: The type of quantity to which the custom module corresponds.
            		- `class_mapping`: A dictionary mapping the custom module type to
            its corresponding observed module class.
            

    Returns:
        dict: a class mapping from the input custom module to its corresponding
        observed module.

    """
    quant_type = get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    assert type(custom_module) in class_mapping, "did not find corresponding observed " \
        f"module class for {type(custom_module)} in mapping: {class_mapping}"
    return class_mapping[type(custom_module)]

def activation_dtype(qconfig):
    """
    retrieves the data type of the activation layer of a neural network configuration
    (qconfig).

    Args:
        qconfig (activation object, returned by calling the `qconfig.activation()`
            method.): quantization configuration for the neural network layer,
            which is used to determine the data type of the activations produced
            by the layer.
            
            		- `qconfig`: This is not None and is an instance of the `pytorch.nn.Module`
            class.
            		- `activation`: This property returns the activation function of the
            module.
            

    Returns:
        `np.dtype`.: the data type of the activation layer's output.
        
        		- The output is an instance of the `torch.utils.checkpoint.Checkpointer`
        class, which represents the activation to be used in the model.
        		- The `dtype` attribute of the output is set to the data type of the
        activation, which can be a numerical value (e.g., `float32`, `float64`)
        or a string (e.g., `'relu', 'sigmoid', etc.'`).
        		- The `requires_grad` attribute of the output is set to `False`, indicating
        that the activation does not support gradient computation.
        

    """
    assert qconfig is not None
    activation = qconfig.activation()
    return activation.dtype

def weight_dtype(qconfig):
    """
    determines the data type of a given neural network configuration's weight.

    Args:
        qconfig (non-void nullable reference to an object of type QConfiguration.):
            QuadTree configuration object that provides the necessary information
            to generate high-quality documentation for the code.
            
            		- `qconfig is not None`: The function assumes that `qconfig` is not
            an empty or missing value, and provides a more informative error message
            in case it is.
            		- `weight = qconfig.weight()`: This line retrieves the `weight`
            attribute of `qconfig`, which is assumed to be a valid property or
            attribute of the object.
            

    Returns:
        `object`.: the data type of the weight attribute of a QuTiP configuration
        object.
        
        		- The output is an instance of `np.ndarray`.
        		- It represents the dtype (data type) of the weights in the quantum circuit.
        		- The dtype is determined by the `qconfig` object passed as input to the
        function.
        

    """
    assert qconfig is not None
    weight = qconfig.weight()
    return weight.dtype

def activation_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized or not, this includes quantizing to quint8, qint8 and qint32 and float16
    """
    return (
        activation_dtype(qconfig) in [
            torch.quint8,
            torch.qint8,
            torch.qint32,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32
        ]
        and (not activation_is_dynamically_quantized(qconfig))
    )

def activation_is_dynamically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    dynamically quantized or not, this includes dynamically quantizing to
    quint8, qint8 and float16
    """
    activation_dtype, _, activation_is_dynamic = \
        get_qconfig_dtypes(qconfig)
    return activation_is_dynamic

def activation_is_int8_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int8 or not, this includes quantizing to quint8, qint8
    """
    return activation_dtype(qconfig) in [torch.quint8, torch.qint8, torch.uint8, torch.int8]

def activation_is_int32_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int32 or not
    """
    return activation_dtype(qconfig) in [torch.qint32, torch.int32]

def weight_is_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be
    quantized or not
    """
    return weight_dtype(qconfig) in [
        torch.quint8,
        torch.qint8,
        torch.float16,
        torch.quint4x2,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32
    ]

def weight_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be statically
    quantized or not
    """
    return weight_dtype(qconfig) in [torch.quint8, torch.qint8, torch.uint8, torch.int8]

def op_is_int8_dynamically_quantized(qconfig) -> bool:
    """ Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
    activation_dtype, weight_dtype, activation_is_dynamic = \
        get_qconfig_dtypes(qconfig)
    return (
        activation_dtype in [torch.quint8, torch.uint8] and
        # for now, the lines below assume fbgemm or qnnpack
        weight_dtype in [torch.qint8, torch.int8] and
        activation_is_dynamic
    )

def get_qconfig_dtypes(qconfig):
    r""" returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_is_dynamic)
    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    act_is_dynamic = getattr(activation, "is_dynamic", False)
    return (activation.dtype, weight.dtype, act_is_dynamic)

def get_quant_type(qconfig):
    """
    determines the type of quantization for a given neural network configuration
    (qconfig) based on the activation and weight dtypes. It returns a QuantType
    value indicating whether the quantization is dynamic, static, or weight-only.

    Args:
        qconfig (non-nullable reference to an object of type `QuantConfig`.):
            quantum config that determines the type of quantization to apply to
            the weights and activation, and is used to determine the appropriate
            QuantType.
            
            		- `qconfig is not None`: The input `qconfig` is not `None`.
            		- `activation()`: Returns the activation function of the quantized
            model.
            		- `weight()`: Returns the weight tensor of the quantized model.
            		- `static_dtypes`: A list of supported static quantization datatypes,
            including `torch.quint8`, `torch.qint8`, `torch.quint4x2`, `torch.qint32`,
            `torch.uint8`, `torch.int8`, `torch.int16`, and `torch.int32`.
            		- `hasattr(activation, 'is_dynamic')`: Checks if the `activation`
            object has an attribute called `'is_dynamic'`.
            		- `is_dynamic`: A boolean indicating whether the activation function
            is dynamic or not.
            		- `dtype in static_dtypes`: Checks if the `weight.dtype` is one of
            the supported static quantization datatypes.
            		- `or otherwise explain its various properties / attributes`: Explains
            the various properties and attributes of `qconfig` if it is not a
            `None` value.
            

    Returns:
        `QuantType`.: a `QuantType` value indicating whether the weight and
        activation are static or dynamic.
        
        		- `QuantType`: This is an enumeration type that represents the type of
        quantization performed on the input data. The possible values are `DYNAMIC`,
        `STATIC`, and `WEIGHT_ONLY`.
        		- `assert qconfig is not None`: This line of code checks that the `qconfig`
        parameter is not none or null, indicating that it is a valid configuration
        object for quantization.
        		- `activation()`: This method returns the activation function associated
        with the input data.
        		- `weight()`: This method returns the weight tensor associated with the
        input data.
        		- `static_dtypes`: This is a list of static dtype values that can be
        used for quantization. These include `torch.quint8`, `torch.qint8`,
        `torch.quint4x2`, `torch.qint32`, `torch.uint8`, `torch.int8`, `torch.int16`,
        and `torch.int32`.
        		- `if weight.dtype in static_dtypes`: This clause checks if the weight
        tensor's dtype is in the list of static dtypes. If it is, then the
        quantization type is determined based on whether the activation function
        is dynamic or not.
        		- `elif activation.dtype in static_dtypes`: This clause checks if the
        activation function's dtype is in the list of static dtypes. If it is,
        then the quantization type is determined based on whether the weight
        tensor's dtype is static or not.
        		- `else`: This clause is executed when neither the weight nor activation
        tensor's dtype is in the list of static dtypes. In this case, the quantization
        type is `WEIGHT_ONLY`.
        		- `raise Exception`: This line of code raises an exception if the
        combination of activation and weight tensors' dtypes is unrecognized. The
        message provided includes the dtypes of the activation and weight tensors.
        

    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32, torch.uint8, torch.int8, torch.int16, torch.int32]
    if weight.dtype in static_dtypes:
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype in static_dtypes:
            return QuantType.STATIC
        else:
            return QuantType.WEIGHT_ONLY

    if weight.dtype == torch.float16:
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype == torch.float16:
            return QuantType.STATIC

    raise Exception(f"Unrecognized dtype combination in get_quant_type: activation({activation.dtype}),"
                    f"weight({weight.dtype})")

def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    """ Checks if the given minimum and maximum values are valid, meaning that
    they exist and the min value is less than the max value.
    """
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(
            "must run observer before calling calculate_qparams. " +
            "Returning default values."
        )
        return False

    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float("inf") and max_val == float("-inf"):
            warnings.warn(
                "must run observer before calling calculate_qparams. " +
                "Returning default values."
            )

            return False

        assert min_val <= max_val, f"min {min_val} should be less than max {max_val}"
    else:
        assert torch.all(
            min_val <= max_val
        ), f"min {min_val} should be less than max {max_val}"

    return True


def calculate_qmin_qmax(quant_min: int, quant_max: int, has_customized_qrange: bool, dtype: torch.dtype,
                        reduce_range: bool) -> Tuple[int, int]:
    r"""Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
    # TODO(jerryzh): Figure out why custom quant_min/quant_max are still adjusted.
    if has_customized_qrange:
        # This initialization here is to be resolve TorchScript compilation issues and allow
        # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
        # The actual values of initial_qmin and initial_qmax will be reset below.
        if dtype in [torch.qint32, torch.int32]:
            initial_quant_min, initial_quant_max = 0, 2**32 - 1
        else:
            initial_quant_min, initial_quant_max = 0, 255
        # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
        # attribute from Optional valid integers for use, based on TorchScript's requirements.
        custom_quant_min, custom_quant_max = quant_min, quant_max
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (
                custom_quant_min,
                custom_quant_max,
            )

        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype in [torch.qint8, torch.int8]:
            assert (
                0 < qrange_len <= 256
            ), "quantization range should be positive and not exceed the maximum bit range (=256)."
        elif dtype in [torch.qint32, torch.int32]:
            assert (
                0 < qrange_len <= 2**32
            ), "quantization range should be positive and not exceed the maximum bit range (=4294967296)."
        if reduce_range:
            quant_min, quant_max = quant_min // 2, quant_max // 2
    else:
        # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
        if dtype in [torch.qint8, torch.int8]:
            if reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        elif dtype in [torch.quint8, torch.uint8]:
            if reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        elif dtype in [torch.qint32, torch.int32]:
            quant_min, quant_max = -1 * (2 ** 31), (2 ** 31) - 1
        else:
            quant_min, quant_max = 0, 15
    return quant_min, quant_max


def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    """
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

def has_no_children_ignoring_parametrizations(module):
    """
    Checks if module._modules is empty or
    if module is a parametrization, checks that module._modules only has
    the 'parametrizations' module
    """
    if len(module._modules) == 0:
        return True
    elif is_parametrized(module):
        return len(module._modules) == 1 and 'parametrizations' in module._modules
    else:
        return False

def _get_path_of_module(root: torch.nn.Module, submodule: torch.nn.Module) -> Optional[str]:
    """ Get the path (fully qualified name) of a submodule

    Example::

    >> class M(torch.nn.Module):
           def __init__(self):
               self.linear = torch.nn.Linear(5, 5)
           def forward(self, x):
               return self.linear(x)

    >> m = M()
    >> l = m.linear
    >> _get_path_of_module(m, l)
    "linear"
    """
    for n, p in root.named_modules():
        if submodule is p:
            return n
    return None

def _get_signature_locals(f: Callable, loc: Dict[str, Any]) -> Dict[str, Any]:
    """ Get local keyword arguments

    Example::

    >> def f(self, a, b=9):
           pass
    >> loc = {"a": 6, "c": 7}
    >> _get_signature_locals(f, loc)
    {"a": 6}
    """
    return {k: v for k, v in loc.items() if k in signature(f).parameters}

def _get_default_kwargs(f: Callable) -> "OrderedDict[str, Any]":
    """ Get all default keyword arguments from function signature

    Example::

    >> def f(self, a, b=9):
           pass
    >> _get_default_kwargs(f)
    {"b": 9}
    """
    kwargs = {}
    for name, param in signature(f).parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default
        elif param.kind is param.VAR_POSITIONAL:
            kwargs[name] = ()
        elif param.kind is param.VAR_KEYWORD:
            kwargs[name] = {}
    return OrderedDict(kwargs)

def _normalize_kwargs(func: Callable, loc: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """ Given a function and local function arguments, normalize the keyword
    arguments by filling in default arguments from function signature

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> loc = {"key2": 6}
    >> _normalize_kwargs(f, loc)
    {"key1": 3, "key2": 6}
    """
    default_kwargs = _get_default_kwargs(func)
    local_kwargs = _get_signature_locals(func, loc)
    normalized_kwargs = default_kwargs.copy()
    for attr, val in local_kwargs.items():
        if attr in normalized_kwargs:
            # override the default keyword arguments
            normalized_kwargs[attr] = val
    return normalized_kwargs

def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    r"""Validates that the user-specified quantization range is properly initialized
    and within the given bound supported by the observer dtype.

    To accommodate lower-bit quantization with respect to the existing torch.qint8 and
    torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
    in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
    values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
    fake quantization. These estimates are compared against parameters learned through backpropagation.
    The related literatures for scale and zero point via backpropagation are as follows:

    Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
    Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
    """
    # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
    # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
    assert (
        quant_min <= 0 <= quant_max
    ), "Used-specified quantization range must include 0."
    assert (
        quant_min < quant_max
    ), "qmin must be strictly less than qmax for user-specified quantization range."


# Functionally equivalent to '_calculate_qparams' in observer.py. Observers must be torchscriptable however and qscheme
# as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
# to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code seems unlikey to change
# (last update over 1 year ago) and when torchscript is fully deprecated we can refactor. TODO(jakeszwe, jerryzh168)
def determine_qparams(
        min_val: torch.Tensor, max_val: torch.Tensor, quant_min: int, quant_max: int,
        dtype: torch.dtype, eps: torch.Tensor, has_customized_qrange: bool,
        qscheme: torch.qscheme = torch.per_tensor_affine) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the quantization parameters, given min and max
    value tensors. Works for both per tensor and per channel cases

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel

    Returns:
        scales: Scales tensor of shape (#channels,)
        zero_points: Zero points tensor of shape (#channels,)
    """
    if not check_min_max_valid(min_val, max_val):
        return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    device = min_val_neg.device
    scale = torch.ones(min_val_neg.size(), dtype=torch.double, device=device)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    if (
        qscheme == torch.per_tensor_symmetric
        or qscheme == torch.per_channel_symmetric
    ):
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        scale = torch.max(scale, eps)
        if dtype in [torch.uint8, torch.quint8]:
            if has_customized_qrange:
                # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                zero_point = zero_point.new_full(
                    zero_point.size(), (quant_min + quant_max) // 2
                )
            else:
                zero_point = zero_point.new_full(zero_point.size(), 128)
    elif qscheme == torch.per_channel_affine_float_qparams:
        scale = (max_val - min_val) / float(quant_max - quant_min)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        # We use the quantize function
        # xq = Round(Xf * inv_scale + zero_point),
        # setting zero_point to (-1 * min *inv_scale) we get
        # Xq = Round((Xf - min) * inv_scale)
        zero_point = -1 * min_val / scale
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, eps)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # For scalar values, cast them to Tensors of size 1 to keep the shape
    # consistent with default values in FakeQuantize.
    if len(scale.shape) == 0:
        # TODO: switch to scale.item() after adding JIT support
        scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
    if len(zero_point.shape) == 0:
        # TODO: switch to zero_point.item() after adding JIT support
        zero_point = torch.tensor(
            [int(zero_point)], dtype=zero_point.dtype, device=device
        )
        if qscheme == torch.per_channel_affine_float_qparams:
            zero_point = torch.tensor(
                [float(zero_point)], dtype=zero_point.dtype, device=device
            )

    return scale.to(torch.double), zero_point.to(torch.int64)

def _get_num_pos_args(f: Callable) -> int:
    """ Get number of positional args for a function

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> _get_num_pos_args(f)
    3
    """
    return len(getfullargspec(f).args)

def get_fqn_to_example_inputs(
    model: torch.nn.Module,
    example_inputs: Tuple[Any, ...]
) -> Dict[str, Tuple[Any, ...]]:
    """ Given a model and its example inputs, return a dictionary from
    fully qualified name of submodules to example_inputs for that submodule,
    e.g. {"linear1": (tensor1,), "linear2": (tensor2,), "sub": (tensor3,),
          "sub.linear1": (tensor4,), ...}

    Used to make quantizing submodules easier now that FX Graph Mode Quantization requires
    example inputs.

    Also works for keyword arguments with default values, we would flatten keyword
    arguments as positional arguments and fill in the missing keyword args with default
    values, e.g. if we have a forward function:
    def forward(self, x, key1=3, key2=3):
        ...

    and we call it with self.submodule(x, key2=6)
    we'll get example_inputs: (x, 3, 6)

    user can also override `key1` with positional arguments as well:
    for self.submodule(x, 5, key2=6)
    we'll get: (x, 5, 6)

    variable positional arguments and variable positional keyword arguments in forward
    function are not supported currently, so please make sure no submodules is using
    them.
    """
    root = model
    fqn_to_example_inputs = {}

    def _patched_module_call(self, *args, **kwargs):
        """
        modifies its inputs to match those of the original module call by adding
        or removing arguments based on the difference between the number of
        positional arguments passed to the original call and the number of key-value
        pairs in the `kwargs` dictionary. It then calls the original function with
        these modified inputs.

        Returns:
            `orig_module_call`.: a modified version of the original function call,
            where the inputs to the original function are replaced with a subset
            of the original inputs and additional keyword arguments.
            
            		- `fqn`: The fully qualified name (FQN) of the module, which is not
            None if the function is successful in finding an example input for the
            submodule.
            		- `submodule_example_inputs_tuple`: A tuple containing the example
            inputs for the submodule, which are obtained by extending the normalized
            keyword arguments with the `popitem` method and then converting them
            to a tuple.
            		- `orig_module_call`: The original module call, which is passed as
            an argument to the function and is used to make the actual module call.
            

        """
        submodule_example_inputs = list(args).copy()
        normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
        # minus 1 to skipping counting `self`
        num_args = _get_num_pos_args(self.forward) - 1
        num_to_pop = num_args - len(submodule_example_inputs)
        while num_to_pop and normalized_kwargs:
            normalized_kwargs.popitem(last=False)
            num_to_pop -= 1
        submodule_example_inputs.extend(normalized_kwargs.values())
        submodule_example_inputs_tuple = tuple(submodule_example_inputs)
        fqn = _get_path_of_module(root, self)
        if fqn is not None:
            fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
        return orig_module_call(self, *args, **kwargs)

    orig_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _patched_module_call  # type: ignore[method-assign]
    try:
        model(*example_inputs)
    finally:
        # restore the module call even if there is an exception
        torch.nn.Module.__call__ = orig_module_call  # type: ignore[method-assign]
    return fqn_to_example_inputs

__all__ = [
    "NodePattern",
    "Pattern",
    "MatchAllNode",
    "check_node",
    "get_combined_dict",
    "is_per_tensor",
    "is_per_channel",
    "getattr_from_fqn",
    "get_qparam_dict",
    "get_swapped_custom_module_class",
    "activation_dtype",
    "weight_dtype",
    "activation_is_statically_quantized",
    "activation_is_dynamically_quantized",
    "activation_is_int8_quantized",
    "activation_is_int32_quantized",
    "weight_is_quantized",
    "weight_is_statically_quantized",
    "op_is_int8_dynamically_quantized",
    "get_qconfig_dtypes",
    "get_quant_type",
    "check_min_max_valid",
    "calculate_qmin_qmax",
    "has_no_children_ignoring_parametrizations",
    "get_fqn_to_example_inputs",
    "to_underlying_dtype",
    "determine_qparams",
    "validate_qmin_qmax",
]
