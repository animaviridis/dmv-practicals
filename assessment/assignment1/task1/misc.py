import inspect


def check_types(func):
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.parameters
        empty = inspect.Signature.empty

        bound_args = sig.bind(*args, **kwargs).arguments

        for arg_name in bound_args:
            if bound_args[arg_name] is None:
                continue

            if params[arg_name].annotation == empty:
                continue

            if not isinstance(bound_args[arg_name], params[arg_name].annotation):
                raise TypeError(f"parameter '{arg_name}': expected {params[arg_name].annotation}, "
                                f"got {type(bound_args[arg_name])}")

        return func(*args, **kwargs)

    return wrapper
