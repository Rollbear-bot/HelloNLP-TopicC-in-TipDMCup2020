# -*- coding: utf-8 -*-
# @Time: 2020/3/28 11:48
# @Author: Rollbear
# @Filename: test_label_pick.py
# 测试附件一（三级标签）的抓取

from entity.label import LabelNode
from util.xl_read import *
import openpyxl


def main():
    # 两种方式：
    # 从excel表格获取所有标签，组成树形结构
    root = read_labels("../xls/e1.xlsx")
    # 从excel表格读取标签，组成列表
    rows = read_labels_by_lines("../xls/e1.xlsx")


if __name__ == '__main__':
    main()
