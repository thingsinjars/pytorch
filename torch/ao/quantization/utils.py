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
    determines whether a given Python node represents a call to a function, method,
    or module, based on its operation and target.

    Args:
        node (`ast.Node`.): nodes to be checked for callable functions or methods.
            
            		- `op`: The operation performed by the node, which is specified in
            the string `op`.
            		- `target`: The target of the operation, which can be a function
            name, method name, or module name.
            		- `modules`: A list of modules that are being called, accessed using
            the variable `modules[str(node.target)]`.
            
            	These properties are used in the function to determine whether the
            node is a call to a function, method, or module, and to return a tuple
            of booleans indicating whether each of these types of calls is present
            in the node.
            
        modules (dict): list of modules that are available for call_module operation
            in the given node.

    Returns:
        bool: a list of booleans indicating whether each operation can be performed
        on the given node.

    """
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    return is_call_function, is_call_method, is_call_module

def get_combined_dict(default_dict, additional_dict):
    """
    combines two dictionaries by copying the contents of `default_dict` and updating
    it with the values from `additional_dict`, returning a new dictionary with the
    combined values.

    Args:
        default_dict (dict): initial dictionary that will be combined with the `additional_dict`.
        additional_dict (dict): 2nd dict that gets merged with the default dict
            to form the final combined dict.

    Returns:
        dict: a combined dictionary containing the values from both the default
        and additional dictionaries.

    """
    d = default_dict.copy()
    d.update(additional_dict)
    return d

def is_per_tensor(qscheme):
    """
    checks if a given tensor scheme is either affine or symmetric per tensor. It
    returns a boolean value indicating whether the scheme matches one of these types.

    Args:
        qscheme (tensor.): transformation scheme for which the method checks if
            it is per-tensor affine or symmetric.
            
            		- `torch.per_tensor_affine`: This indicates that the tensor is affine
            transformable, meaning it can be transformed using a single affine transformation.
            		- `torch.per_tensor_symmetric`: This indicates that the tensor is
            symmetric, meaning it has the same properties before and after being
            multiplied by its inverse.
            

    Returns:
        bool: a boolean value indicating whether the given query scheme is per-tensor
        affine or symmetric.

    """
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def is_per_channel(qscheme):
    """
    checks if a given scheme is one of the supported per-channel transformations
    in PyTorch, including affine and symetric transformations with float-point parameters.

    Args:
        qscheme (ndarray or tensor.): query scheme to check if it is supported by
            the `is_per_channel` function, which returns `True` if the query scheme
            is one of the specified ones.
            
            		- `torch.per_channel_affine`: This is an instance of `torch.nn.Module`,
            which means it has a set of parameters that define a linear transformation
            to be applied to each channel of the input tensor.
            		- `torch.per_channel_affine_float_qparams`: This is a special type
            of `torch.per_channel_affine` instance that includes floating-point
            parameter values for the affine transformation.
            		- `torch.per_channel_symmetric`: This is an instance of `torch.nn.Module`
            that applies a symmetric affine transformation to each channel of the
            input tensor.
            

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
    maps a Quartus DataType (qdtype) to its underlying Python data type, checking
    for supported types and returning the mapped value.

    Args:
        qdtype (int): ndarray's underlying dtype, which is used to map the ndarray
            to an equivalent dtype that is supported by the function.

    Returns:
        int: the underlying dtype of the given quaternion dimensional type.

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
    generates a dictionary of quantization parameters for a given tensor or
    placeholder observer. It determines the appropriate quantization scheme based
    on the input observer's qscheme and returns the scale, zero point, and other
    quantization parameters.

    Args:
        observer_or_fake_quant (`torch.ao.quantization.observer.PlaceholderObserver`
            or any other object that has `dtype`, `qscheme`, `calculate_qparams`,
            `quant_min`, and `quant_max` attributes.): observer or fake quantizer
            that provides the necessary parameters for generating high-quality
            documentation for its corresponding code.
            
            		- `qscheme`: The quantization scheme used for the observer or fake
            quant. It can be a string (e.g., "per-tensor", "per-channel") or an
            instance of `torch.ao.quantization.QuantizationScheme`. If it's not
            specified, it defaults to None.
            		- `dtype`: The data type of the observer or fake quant. It can be a
            string (e.g., "float32", "int8") or an instance of `torch.utils.data.Dataset`.
            If it's not specified, it defaults to the dtype of the input tensor.
            		- `axis`: The axis along which the quantization is performed for
            per-channel quantization scheme. It can be a scalar value or a vector
            representing the axis(es) in case of per-tensor affine scheme. If it's
            not specified, it defaults to -1 (the default axis for per-tensor affine).
            		- `calculate_qparams`: A method that calculates the scale and zero
            point parameters for the quantization. The method can be a scalar value
            or a function returning a tuple of two scalars. If it's not specified,
            it defaults to the `calculate_qparams` method of the `torch.ao.quantization.Observer`.
            		- `quant_min`: The minimum possible value for the quantized tensor.
            It can be a scalar value or a function returning a scalar value. If
            it's not specified, it defaults to the minimum possible value for the
            input tensor.
            		- `quant_max`: The maximum possible value for the quantized tensor.
            It can be a scalar value or a function returning a scalar value. If
            it's not specified, it defaults to the maximum possible value for the
            input tensor.
            

    Returns:
        dict: a dictionary containing parameters for quantizing a tensor or fake
        quantum observer, including the qscheme, dtype, axis, scale, zero_point,
        and quantization min/max values.

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
    maps a given custom module to its corresponding observed module class based
    on a provided mapping, and returns the mapped module class.

    Args:
        custom_module (object.): custom module for which the corresponding observed
            module class needs to be found in the `custom_module_class_mapping`.
            
            		- `custom_module`: The deserialized custom module instance that is
            passed as an argument to the function. Its type is determined by the
            value of the `quant_type` variable, which is obtained from the `qconfig`
            parameter.
            		- `custom_module_class_mapping`: A mapping of quant types to their
            corresponding observed module classes. This mapping is used to look
            up the appropriate observed module class for the given quant type.
            
        custom_module_class_mapping (mapping of type `quant_type` to class object.):
            mapping between observed module types and their corresponding classes,
            which is used to locate the appropriate class for a given observed
            module type.
            
            		- `custom_module`: The custom module that needs to be swapped with
            a corresponding observed module. (type: any)
            		- `custom_module_class_mapping`: A mapping of quant types to their
            corresponding observed module classes. (type: dict)
            		- `qconfig`: The quant configuration object containing the quant
            type for which the module class needs to be retrieved. (type: qconfig)
            
        qconfig (dict): configuration of the quantization module, which is used
            to determine the appropriate class mapping for the custom module based
            on its type.

    Returns:
        mapping containing the correspondence between quantitative types and
        observed module classes.: a module class mapping for the given custom
        module and quant type.
        
        		- `quant_type`: This variable represents the quantum type associated
        with the custom module class mapping. It is obtained from the `qconfig`
        parameter passed to the function.
        		- `class_mapping`: This variable contains the mapping between the observed
        module and its corresponding quantum type. The value of this variable is
        a dictionary, where the keys are the observed module classes and the values
        are the corresponding quantum types. If the `custom_module` class is not
        found in the mapping, an error message is raised.
        		- `type(custom_module)`: This represents the type of the custom module
        passed as input to the function. It is used to determine the corresponding
        quantum type from the `class_mapping` dictionary.
        

    """
    quant_type = get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    assert type(custom_module) in class_mapping, "did not find corresponding observed " \
        f"module class for {type(custom_module)} in mapping: {class_mapping}"
    return class_mapping[type(custom_module)]

def activation_dtype(qconfig):
    """
    returns the data type of the activation value associated with a given quantum
    circuit configuration.

    Args:
        qconfig (activation object returned by its corresponding configuration
            function call.): quantum circuit configuration for which the activation
            type is determined.
            
            		- `qconfig`: A non-null input indicating the activation type to be
            performed on the output of the transformed query.
            

    Returns:
        `np.dtype`.: the data type of the activation output of a neural network,
        as determined by the configuration provided in the input `qconfig`.
        
        		- The activation function that is used to compute the output for each
        element in the input tensor. (Eg., `ReLU`, `Sigmoid`, etc.)
        		- The data type of the output elements after computing the activation
        function. This can be any of the supported data types by TensorFlow, such
        as `tf.int32`, `tf.float32`, or others.
        

    """
    assert qconfig is not None
    activation = qconfig.activation()
    return activation.dtype

def weight_dtype(qconfig):
    """
    returns the data type of a given weight value.

    Args:
        qconfig (ndarray, according to the supplied code snippet.): quantum circuit
            configuration that provides the weight of the quantum register.
            
            		- `qconfig`: A non-null reference to an instance of the `QuantumConfig`
            class.
            

    Returns:
        `np.dtype`.: the data type of the weight parameter.
        
        	1/ `assert qconfig is not None`: This line verifies that the input `qconfig`
        is not None before proceeding with the function.
        	2/ `weight = qconfig.weight()`: This line extracts the `weight` attribute
        from the `qconfig` object using the `.weight()` method.
        	3/ `return weight.dtype`: This line returns the `dtype` of the extracted
        `weight` attribute.
        

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
    determines the type of quantization to apply to a neural network based on the
    activation and weight dtypes. It returns `QuantType.DYNAMIC`, `QuantType.STATIC`,
    or `QuantType.WEIGHT_ONLY` depending on the combination of activation and
    weight dtypes.

    Args:
        qconfig (non-nullable reference to an object.): quantization configuration
            for the neural network, which contains information about the activation
            and weight types to be used for quantization.
            
            		- `qconfig.activation()`: Gets the activation function of the quantized
            model.
            		- `qconfig.weight()`: Gets the weight tensor of the quantized model.
            		- `static_dtypes`: A list of static data types that are supported
            for quantization. These include [torch.quint8, torch.qint8, torch.quint4x2,
            torch.qint32, torch.uint8, torch.int8, torch.int16, torch.int32].
            		- `hasattr(activation, 'is_dynamic')`: Checks if the `activation`
            object has an attribute called `'is_dynamic'`. If it does, then the
            activation function is dynamic.
            		- `hasattr(activation, 'dtype')`: Checks if the `activation` object
            has an attribute called `'dtype'`. If it does, then the activation
            function is of a supported type.
            
            	If any of these conditions are met, the function returns a `QuantType`
            value indicating the type of quantization that can be applied to the
            model. Otherwise, an error message is raised.
            

    Returns:
        `QuantType`.: a `QuantType` value indicating whether the Quantizer's weights
        and activations are dynamic or static.
        
        		- `QuantType`: This is an enumeration type that represents different
        types of quantization for neural networks. The possible values are `DYNAMIC`,
        `STATIC`, and `WEIGHT_ONLY`.
        		- `activation`: This is a property of the input `qconfig` object, which
        represents the activation function used in the neural network. Its value
        can be one of the allowed values (`torch.nn.functional.ReLU`,
        `torch.nn.functional.Sigmoid`, etc.).
        		- `weight`: This is a property of the input `qconfig` object, which
        represents the weight tensor of the neural network. Its value can be one
        of the allowed values (`torch.Tensor`, `torch.nn.parameter.ParameterTuple`,
        etc.).
        		- `static_dtypes`: This is a list of allowed static quantization dtypes
        for neural networks. The values in this list are `torch.quint8`, `torch.qint8`,
        `torch.quint4x2`, `torch.qint32`, `torch.uint8`, `torch.int8`, `torch.int16`,
        and `torch.int32`.
        		- `hasattr(activation, 'is_dynamic')`: This is a boolean property that
        indicates whether the activation function has the `is_dynamic` attribute.
        If it does, then the quantization type is dynamic.
        		- `hasattr(activation, 'dtype')`: This is a boolean property that indicates
        whether the activation function has the `dtype` attribute. If it does,
        then the quantization type is static.
        

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
        modifies its call to forward a list of normalized arguments and key-value
        pairs to a submodule's `forward` method, while skipping over `self`. It
        then returns the result of the original call.

        Returns:
            `OrigModuleCall`.: a modified version of the original module call,
            where some of the positional arguments have been replaced with values
            from the `normalized_kwargs` dictionary.
            
            		- `fqn`: A Noneable representing the fully qualified name of the
            module being called, or None if no such information is available.
            		- `submodule_example_inputs`: A list of positional arguments passed
            to the called module, obtained by recursively traversing the function's
            call tree and removing any duplicates.
            		- `normalized_kwargs`: A dictionary containing the keyword arguments
            passed to the called module, after normalizing their keys using the
            forwarded mapping from the parent module's `forward` attribute. The
            size of this dictionary indicates the number of non-empty keyword arguments.
            		- `num_args`: An integer representing the total number of positional
            arguments passed to the called module, including any duplicates removed
            from `submodule_example_inputs`.
            		- `num_to_pop`: An integer representing the number of positive
            integers in the list of `num_args` that have not been popped from the
            `submodule_example_inputs` list during iteration.
            
            	These properties are used to determine how many arguments to pop from
            the `submodule_example_inputs` list during each iteration, based on
            the number of non-empty keyword arguments passed to the called module.
            

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
