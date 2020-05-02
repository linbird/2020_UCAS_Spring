from functools import partial
import time
from multiprocessing import Pool
import os


def my_print(x, y):
    print(os.getpid(), ': ', x+y)
    time.sleep(1)
    return x+y

if __name__ == '__main__':# 多线程，多参数，partial版本
    x = [1, 2, 3, 4, 5, 6]
    y = 1
 
    partial_func = partial(my_print, y=y)
    pool = Pool(4)
    result=[]
    result.append(pool.map(partial_func, x))
    pool.close()
    pool.join()
    for i in result:
        print(i)
