# -*- coding: utf-8 -*-
# @Time: 2020/4/23 16:07
# @Author: Rollbear
# @Filename: output.py
# 控制台输出工具


def output_cluster(labels: list, docs):
    """输出聚类结果"""
    # 按标签分类存放在字典中
    comm_cluster = {}
    for index, doc in enumerate(docs):
        if labels[index] in list(comm_cluster.keys()):
            comm_cluster[labels[index]].append(doc)
        else:
            comm_cluster[labels[index]] = [doc]

    # 展示聚类结果
    for label, cluster in comm_cluster.items():
        print(f"Label{label}: ")
        for count, text in enumerate(cluster):
            print(f"Text {count}: " + text)
        print("-------------------------------\n")
