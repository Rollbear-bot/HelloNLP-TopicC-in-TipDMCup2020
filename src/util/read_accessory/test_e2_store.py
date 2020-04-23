# -*- coding: utf-8 -*-
# @Time: 2020/4/9 15:50
# @Author: Rollbear
# @Filename: test_e2_store.py

import sqlite3

from util.xl_read import read_labels_by_lines


def main():
    # 连接数据库
    conn = sqlite3.connect("../resources/test.db")
    cursor = conn.cursor()  # 创建一个用于操作数据库的光标

    # 建立适用于附件二的表
    # 该表含有七个列，分别是
    # 行编号、留言编号、用户编号、留言主题、留言时间、留言详情、一级分类
    cursor.execute('''create table Comments
    (
    	id int not null
    		constraint Comments_pk
    			primary key,
        Comment_id int not null,
    	User_id text not null,
    	Topic text not null,
    	Date_and_Time text not null,
    	Detail text not null,
    	First_Level_Label text not null
    );''')

    # 从附件二的excel表中读取记录，写入到数据库中
    # 读取行
    rows = read_labels_by_lines("../../../resources/xls/e2.xlsx")

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

        cursor.execute(f'''INSERT INTO Comments 
            VALUES ({row_id}, {row[0]}, \'{row[1]}\', \'{row[2]}\', \'{row[3]}\', \'{row[4]}\', \'{row[5]}\');''')
        row_id += 1

    conn.commit()


if __name__ == '__main__':
    main()
