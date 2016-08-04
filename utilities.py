def all_subclasses(cls):
    result = {subclass.__name__: subclass for subclass in cls.__subclasses__()}
    recurse = [all_subclasses(subclass) for subclass in result.values()]
    for r in recurse:
        result.update(r)
    return result
