from inspect import getfullargspec

class Function(object):
    '''Function is a wrapper over standard python function
    '''
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        '''When invoked like a function, it internally invokes
        the wrapped function and returns the returned value.
        '''

        # Fetching the function to be invoked from the virtual namespace
        # throught the arguments
        fn = Namespace.get_instance().get(self.fn, *args)
        if not fn:
            raise Exception('No matching function found!')

        # Invoking the wrapped function and returning the value
        return fn(*args, **kwargs)

    def key(self, args=None):
        '''Returns the key that will uniquely identify a function
        (even when it is overloaded).
        '''
        # If args not specified, extract the args from the function 
        # definition
        if args is None:
            args = getfullargspec(self.fn).args

        return tuple([
            self.fn.__module__,
            self.fn.__class__,
            self.fn.__name__,
            len(args or []),
            # [type(arg) for arg in args],
        ])


class Namespace(object):
    '''Namespace is a singleton class that is responsible
    for holding all the functions.
    '''
    __instance = None

    def __init__(self) -> None:
        if self.__instance is None:
            self.function_map = dict()
            Namespace.__instance = self
        else:
            raise Exception("Cannot instantiate a virtual namespace again!")

    @staticmethod
    def get_instance():
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance

    def register(self, fn):
        '''registers the function in the virtual namespace and returns 
        an instance of callable Function that wraps the function fn.
        '''
        func = Function(fn)
        self.function_map[func.key()] = fn
        return func
    
    def get(self, fn, *args):
        '''get returns the matching function from the virtuak namespace.

        Returns None if it did not find any matching function.
        '''
        func = Function(fn)
        return self.function_map.get(func.key(args=args))



def overload(fn):
    '''overload is the decorator that wraps the function
    and returns a callable object of type Function.
    '''
    return Namespace.get_instance().register(fn)
