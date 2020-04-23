# -*- coding: utf-8 -*-
# @Time: 2020/4/21 23:14
# @Author: Rollbear
# @Filename: named_entity_recognition.py
# 命名实体识别

import jieba
import jieba.posseg as pseg

from util.xl_read import read_xl_by_line


def main():
    jieba.enable_paddle()  # 打开飞桨模式

    data = read_xl_by_line("../resources/xls/e3.xlsx")
    lines = [row[2] + "。" + row[4] for row in data]

    # 要求的词性
    word_type = ["n", "f", "s", "ns", "nt", "nw", "nz",
                 "v", "vd", "vn",
                 "a", "ad", "an", "d",
                 "LOC", "ORG"]

    # 将文本中词性满足挑选条件的词挑选出来
    for line in lines:
        words = pseg.cut(line)
        for pair in [list(p) for p in words]:
            if pair[1] in word_type:
                print(pair[0], pair[1])
        print("--------------------------------")


if __name__ == '__main__':
    main()

