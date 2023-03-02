def dataclass_to_namespace(args, args_name):
    # Dataclass arg to python namespace
    if args.__contains__(args_name):
        for key, value in args.__getattribute__(args_name).__dict__.items():
            args.__setattr__(key, value)
        args.__delattr__(args_name)
    return args
