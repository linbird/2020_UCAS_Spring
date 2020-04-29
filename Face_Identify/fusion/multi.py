# -*- coding:utf-8 -*-

import time
import multiprocessing


def job(x ,y):
    """
        :param x:
        :param y:
        :return:
    """
    return x * y

def job1(z):
    """
        :param z:
        :return:
        """
    return job(z[0], z[1])


if __name__ == "__main__":
    time1=time.time()
    pool = multiprocessing.Pool(2)
    data_list=[(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)]
    res = pool.map(job1,data_list)
    time2=time.time()
    print(res)
    pool.close()
    pool.join()
    print('总共耗时：' + str(time2 - time1) + 's')
