"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""
import functools


def _disable_dynamo(fn=None, recursive=True):
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    if fn is not None:

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            """
            provides a mechanism to disable Dynamo's function recursion. It takes
            an arbitrary function `fn` and any number of positional or keyword
            arguments, and applies Dynamo's disable() method to it before executing
            the function with those arguments.

            Returns:
                instance of the Torch Dynamo Disable Method.: the result of calling
                the `disable` method on a function and passing it any required
                arguments or keyword arguments.
                
                	The `import torch._dynamo` statement imports the `_dynamo` module
                from the Torch library. This module provides an interface to the
                DynamoVM runtime, which is a Just-In-Time (JIT) compiler and
                execution engine for Torch tensors.
                
                	The `disable(fn, recursive)` function call disables the default
                behavior of the `fn` function, and recursively enables it for any
                nested functions. This allows for dynamic control over the behavior
                of the function, enabling more advanced use cases such as code
                generation or optimization.
                
                	The `(*args, **kwargs)` arguments are passed to the `disable`
                function, indicating that the function should be disabled for these
                arguments. The `recursive` argument indicates that the disablement
                should be performed recursively on any nested functions.
                

            """
            import torch._dynamo

            return torch._dynamo.disable(fn, recursive)(*args, **kwargs)

        return inner
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(_disable_dynamo, recursive=recursive)
