# -*- coding: utf-8 -*-
# @Time: 2020/4/9 16:26
# @Author: Rollbear
# @Filename: e3_store.py

import sqlite3

from util.xl import read_labels_by_lines


def main():
    # 连接数据库
    conn = sqlite3.connect("../resources/test.db")
    cursor = conn.cursor()  # 创建一个用于操作数据库的光标

    # 读取行
    rows = read_labels_by_lines("../../../resources/xls/e3.xlsx")

    # 将记录插入到数据库
    row_id = 0
    for row in rows:
        if row[0] is None:
            break
        # 去除“留言详情”中的单引号，防止sql注入
        s = row[4]
        while s.find("\'") >= 0:
            char_lt = list(s)
            char_lt.remove("\'")
            s = "".join(char_lt)
        row = list(row)
        row[4] = s

        cursor.execute(f'''INSERT INTO Comments_with_Likes 
            VALUES ({row_id}, {row[0]}, \'{row[1]}\', \'{row[2]}\', \'{row[3]}\',
            \'{row[4]}\', \'{row[5]}\', \'{row[6]}\');''')
        row_id += 1

    conn.commit()


if __name__ == '__main__':
    main()