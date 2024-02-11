import torch
import copy
from typing import Dict, Any

__all__ = [
    "set_module_weight",
    "set_module_bias",
    "get_module_weight",
    "get_module_bias",
    "max_over_ndim",
    "min_over_ndim",
    "channel_range",
    "cross_layer_equalization",
    "equalize",
    "converged",
]

_supported_types = {torch.nn.Conv2d, torch.nn.Linear}
_supported_intrinsic_types = {torch.ao.nn.intrinsic.ConvReLU2d, torch.ao.nn.intrinsic.LinearReLU}
_all_supported_types = _supported_types.union(_supported_intrinsic_types)

def set_module_weight(module, weight) -> None:
    """
    This function sets the weight parameter of a given module to a specified value
    using Torch's ` Parameter` class. It supports a variety of module types and
    assigns the weight to either the module or one of its elements depending on
    the type of module.

    Args:
        module (list): The `module` input parameter is an object that should be a
            nnn module of one of the supported types and the weight to be set to
            it is the second parameter.
        weight (): The weight parameter modifies the module's parameters so that
            it scales the outputs when multiplied by the weights.

    """
    if type(module) in _supported_types:
        module.weight = torch.nn.Parameter(weight)
    else:
        module[0].weight = torch.nn.Parameter(weight)

def set_module_bias(module, bias) -> None:
    """
    The provided code defines a Python function named `set_module_bias`. It takes
    two arguments: `module`, and `bias`. What the function does is that it sets
    the bias for a PyTorch module to a new value passed as argument if type(module)
    exists among the listed supported types or module[0].bias (this will only set
    the bias if it exists).

    Args:
        module (None): Here's a concise answer to your question:
            
            `module` is a Python object that serves as an argument to this function
            set_module_bias(), which may or may not have properties like 'bias',
            given by the `type(module)` test _supported_types.
        bias (float): The input `bias` is being set as a parameter of whatever
            instance of the supported classes has been passed into the `set_module_bias`
            function. The precise purpose is hard to determine with so little
            context (I only have the input function to review), and the types being
            passed cannot be reliably determined either given that we've stripped
            all relevant context; nonetheless I can safely say that `bias` should
            refer to a valid PyTorch Parameter if the module and classes supported
            by this are properly implemented and will become one when assigned.
            Is there anything specific you wish me to address concerning this input
            parameter?

    """
    if type(module) in _supported_types:
        module.bias = torch.nn.Parameter(bias)
    else:
        module[0].bias = torch.nn.Parameter(bias)

def get_module_weight(module):
    """
    This function retrieves the weight of a given PyTorch module. If the module
    is a supported type (i.e., a module implemented by PyTorch), it returns the
    module's weight directly. Otherwise (i.e., the module is a list or tensor),
    it returns the weight of the first element inside the module.

    Args:
        module (list): The `module` input parameter is any Python object and this
            function figures out how to extract its weight.

    Returns:
        float: The function `get_module_weight(module)` returns the weight of the
        first element inside the given module if the type of the module is not
        supported or if the module is not iterable (e.g., a list or tuple). It
        otherwise returns the weight of the module itself.
        
        For example;  if `supported_types`= (_supported_types), the function returns
        either the `weight' of an object belonging to one of those types (`e.g.
        _,int`,list') or ' None'. If `module is not iterable', it returns `'None'`.

    """
    if type(module) in _supported_types:
        return module.weight
    else:
        return module[0].weight

def get_module_bias(module):
    """
    This function returns the bias attribute of a module or its first layer (if
    it's not a supported type).

    Args:
        module (list): The `module` parameter is not actually used anywhere within
            the function.  Instead it takes any valid Python object and defaults
            to its first (0th) index or attribute to check if it has a bias value;
            if that's empty string it will raise a AttributeError.  Thus `module`
            exists only so that this code may support more varieties of input than
            those which directly have the desired property or attribute of `.bias`,
            but any such type of object (including nested modules like list items)
            with some type of '`.bias'` item index qualifies for its return.

    Returns:
        float: The function `get_module_bias()` returns a single bias value.

    """
    if type(module) in _supported_types:
        return module.bias
    else:
        return module[0].bias

def max_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.max' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input

def min_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.min' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)
    return input

def channel_range(input, axis=0):
    """Find the range of weights associated with a specific channel."""
    size_of_tensor_dim = input.ndim
    axis_list = list(range(size_of_tensor_dim))
    axis_list.remove(axis)

    mins = min_over_ndim(input, axis_list)
    maxs = max_over_ndim(input, axis_list)

    assert mins.size(0) == input.size(axis), "Dimensions of resultant channel range does not match size of requested axis"
    return maxs - mins

def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    """Scale the range of Tensor1.output to equal Tensor2.input.

    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    if type(module1) not in _all_supported_types or type(module2) not in _all_supported_types:
        raise ValueError("module type not supported:", type(module1), " ", type(module2))

    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)

    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError("Number of output channels of first arg do not match \
        number input channels of second arg")

    bias = get_module_bias(module1)

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # producing scaling factors to applied
    weight2_range += 1e-9
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    bias = bias * inverse_scaling_factors

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    # pads axis to (1D) tensors to then be broadcasted
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)

    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    set_module_weight(module1, weight1)
    set_module_bias(module1, bias)
    set_module_weight(module2, weight2)

def equalize(model, paired_modules_list, threshold=1e-4, inplace=True):
    """Equalize modules until convergence is achieved.

    Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    Implementation of this referced section 4.1 of this paper https://arxiv.org/pdf/1906.04721.pdf

    Args:
        model: a model (nn.module) that equalization is to be applied on
        paired_modules_list: a list of lists where each sublist is a pair of two
            submodules found in the model, for each pair the two submodules generally
            have to be adjacent in the model to get expected/reasonable results
        threshold: a number used by the converged function to determine what degree
            similarity between models is necessary for them to be called equivalent
        inplace: determines if function is inplace or not
    """
    if not inplace:
        model = copy.deepcopy(model)

    name_to_module : Dict[str, torch.nn.Module] = {}
    previous_name_to_module: Dict[str, Any] = {}
    name_set = {name for pair in paired_modules_list for name in pair}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    while not converged(name_to_module, previous_name_to_module, threshold):
        for pair in paired_modules_list:
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

    return model

def converged(curr_modules, prev_modules, threshold=1e-4):
    """Test whether modules are converged to a specified threshold.

    Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the difference
    between the associated modules in each dictionary

    """
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError("The keys to the given mappings must have the same set of names of modules")

    summed_norms = torch.tensor(0.)
    if None in prev_modules.values():
        return False
    for name in curr_modules.keys():
        curr_weight = get_module_weight(curr_modules[name])
        prev_weight = get_module_weight(prev_modules[name])

        difference = curr_weight.sub(prev_weight)
        summed_norms += torch.norm(difference)
    return bool(summed_norms < threshold)
