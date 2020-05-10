# -*- coding: utf-8 -*-
# @Time: 2020/5/3 9:51
# @Author: Rollbear
# @Filename: data_analysis.py
# 附件数据分析

from util.dataset import *


def main():
    stop = fetch_default_stop_words()
    name_lt = ["full_dataset_sheet_2", "full_dataset_sheet_3", "full_dataset_sheet_4"]
    for ds in name_lt:
        show_data_analysis(ds_name=ds, cut_all=False, stop_words=stop)


if __name__ == '__main__':
    main()
