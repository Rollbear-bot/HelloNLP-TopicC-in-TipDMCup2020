# -*- coding: utf-8 -*-
# @Time: 2020/5/8 9:48
# @Author: Rollbear
# @Filename: train_knn.py
# 训练kNN分类器的脚本

import gensim
from gensim.models import Word2Vec
from sklearn.externals import joblib  # 模型保存与加载
from sklearn.neighbors import KNeighborsClassifier  # 最近邻分类器

from util.dataset import *
from util.path import *
from util.vec import doc_vec


def main():
    # 停用词表
    stop = fetch_default_stop_words()
    # 读取附件二
    comments = read_xl_by_line(sheet_2_input)
    comm_dict_2 = fetch_data("full_dataset_sheet_2", stop_words=stop,
                             cut_all=True, remove_duplicates=False)

    # 加载word2vec模型
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=False)

    # 表2中涉及的所有一级标签
    target_names = list(set([row[5] for row in comments]))
    # 将标签名写入到文件
    with open(knn_model_target_names, "w", encoding="utf8") as f:
        f.writelines(target_names)
    # 将表2中标注的所有一级标签转化成数字表示（target_names中的index）
    targets = [target_names.index(row[5]) for row in comments]

    # 计算所有评论的文档向量
    comms_vec = [doc_vec(
        comm_dict_2[row[0]].seg_topic + comm_dict_2[row[0]].seg_detail,
        wv_model)
        for row in comments]

    # 引入KNN模型（这里设置的参数是经过贝叶斯优化得出的一组参数）
    knn = KNeighborsClassifier(leaf_size=11,
                               n_neighbors=13)

    # 训练knn模型
    knn.fit(comms_vec, targets)
    # 将得到的knn模型保存到预设的路径
    joblib.dump(knn, knn_model_path)


if __name__ == '__main__':
    main()
