from functools import wraps

def coroutine(func):
    """
    데커레이터, 'func'를 기동해서 첫 번째 'yield'까지 진행한다.
    """
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return primer


if __name__ == '__main__':
    from inspect import getgeneratorstate
    
    @coroutine
    def averager():
        total = 0.0
        count = 0
        average = None
        while True:
            term = yield average
            total += term
            count += 1
            average = total/count

    coro_avg = averager()
    print(getgeneratorstate(coro_avg))  # GEN_CREATED, @coroutine에 의해서 시작
    print(coro_avg.send(10))
    print(coro_avg.send(20))
    print(coro_avg.send(15))

    print('done')