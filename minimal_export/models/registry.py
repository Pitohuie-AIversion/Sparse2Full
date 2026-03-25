MODEL_REGISTRY = {}

def register_model(name, aliases=None):
    """Decorator to register a model."""
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        if aliases:
            for alias in aliases:
                MODEL_REGISTRY[alias] = cls
        return cls
    return wrapper

def create_model(name, **kwargs):
    """Factory method to create a model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry.")
    return MODEL_REGISTRY[name](**kwargs)
