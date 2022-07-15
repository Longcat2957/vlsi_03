import time
from functools import wraps

def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        before = time.time()
        value = function(*args, **kwargs)
        fname = function.__name__
        print(f'{fname} took {(time.time() - before)*1000:.5f}ms to execute')
        return value
    return wrapper

if __name__ == '__main__':
    
    @timer
    def myfunction(x):
        result = 1
        for i in range(1, x):
            result *= 1
        return result
    
    myfunction(100)