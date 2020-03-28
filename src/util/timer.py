# -*- coding: utf-8 -*-
# @Time: 2020/3/28 16:51
# @Author: Rollbear
# @Filename: timer.py
# 计时器

from functools import wraps
from time import time


def timer(func):
    """计时器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        func_res = func(*args, **kwargs)
        # 打印函数执行用时
        print(f"exec \"{func.__name__}\" in {time() - start}s.")
        return func_res  # 返回函数运行结果
    return wrapper
