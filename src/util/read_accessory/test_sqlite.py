# -*- coding: utf-8 -*-
# @Time: 2020/3/28 16:12
# @Author: Rollbear
# @Filename: test_sqlite.py

import sqlite3

from util.xl import read_labels_by_lines


def main():
    # todo::封装成数据库方法
    table_name = "Labels"
    rows = read_labels_by_lines("../../../resources/xls/e1.xlsx")

    # 使用sqlite数据库将标签持久化
    conn = sqlite3.connect("../resources/test.db")
    cursor = conn.cursor()  # 创建一个用于操作数据库的光标

    # 执行sql来插入记录
    label_id = 0
    for row in rows:
        cursor.execute(f'''INSERT INTO {table_name} 
            VALUES ({label_id}, \'{row[0]}\', \'{row[1]}\', \'{row[2]}\');''')
        label_id += 1

    conn.commit()  # 提交数据


if __name__ == '__main__':
    main()
