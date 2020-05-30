from functools import wraps


def property_cached(func):
    """
    Decorator for replacing @property. The result is cached onto the parent object under a `cache` dict attribute.
    Subsequent calls to the property retrieve the cached value rather than recomputing the method.
    """
    @wraps(func)
    def with_cache(self):
        cache_name = func.__name__
        # Instantiate cache dict if not yet defined
        if not hasattr(self, "cache"):
            object.__setattr__(self, "cache", {})
        if cache_name not in self.cache:
            self.cache[cache_name] = func(self)
        return self.cache[cache_name]

    return property(with_cache)
