# -*- coding: utf-8 -*-
# @Time: 2020/3/28 11:49
# @Author: Rollbear
# @Filename: test_jieba.py
# 中文分词包jieba测试

import jieba


def main():
    s = "我来自台中。"
    jieba.add_word("台中")
    res = jieba.cut(s)  # cut方法返回一个迭代器
    for word in res:
        print(word)


if __name__ == '__main__':
    main()
