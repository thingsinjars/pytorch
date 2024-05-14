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
            disable the dynamic computation graph recording for a given function
            `fn` and its nested calls recursively by wrapping them with
            `torch._dynamo.disable(fn, recursive)`.

            Returns:
                `torch.Tensor`.: the result of calling the provided function `fn`
                with the passed arguments and keyword arguments after disabling
                dynamic computation tracing using `torch._dynamo.disable(fn, recursive)`.
                
                		- The function returns a tuple containing the result of calling
                the provided function `fn` with the passed arguments `*args` and
                keyword arguments `**kwargs`.
                		- The returned value is a dynamo-enabled version of the original
                function call.
                		- The `torch._dynamo` module is imported to enable dynamo-based
                tracing of the function.

            """
            import torch._dynamo

            return torch._dynamo.disable(fn, recursive)(*args, **kwargs)

        return inner
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(_disable_dynamo, recursive=recursive)
