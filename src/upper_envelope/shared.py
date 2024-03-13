import functools
import inspect


def process_function_args_to_kwargs(func):
    signature = set(inspect.signature(func).parameters)

    @functools.wraps(func)
    def processed_func(**kwargs):
        func_kwargs = {key: kwargs[key] for key in signature}

        return func(**func_kwargs)

    return processed_func
