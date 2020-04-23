# -*- coding: utf-8 -*-
# @Time: 2020/4/21 23:14
# @Author: Rollbear
# @Filename: named_entity_recognition.py
# 命名实体识别、词性标注

import pandas as pd  # 矩阵处理工具
from gensim.models import Word2Vec  # Word2Vec词嵌入模型
from sklearn.cluster import MeanShift  # 均值漂移模型

from util.dataset import fetch_data
from util.dataset import fetch_default_stop_words  # 加载停用词的方法
from util.output import output_cluster
from util.vec import doc_vec_with_weight  # 计算文档向量的方法
from util.word_type import pick_specific_type_words  # 选择特定词性词语的方法
from util.xl_read import read_xl_by_line  # 读取xlsx表格的方法


def main():
    rows = read_xl_by_line("../resources/xls/e3.xlsx")  # 以元组的形式加载附件表
    # todo::聚类时是否使用“留言主题”？
    lines = [row[2] + "。" + row[4] for row in rows]

    stop = fetch_default_stop_words()  # 默认停用词
    stop.extend(["\t", "\n", ""])  # 附加的停用词

    # 要求的特定词性
    word_type = ["n", "s", "ns", "nt", "nw", "nz",  # 各种名词
                 "a", "an",  # 形容词
                 "LOC", "ORG"]  # 专有地名、组织机构名

    # 挑出指定词性的词，将他们放在列表中（每个留言文本是一个词列表）
    specific_only = pick_specific_type_words(text=lines, types=word_type, stop_words=stop)

    # --------------------------------------
    # 对特定词性的词加权后进行文本聚类

    # 加载之前训练的Word2Vec模型
    # wv_model = gensim.models.KeyedVectors.load_word2vec_format(
    #     "../resources/word2vec_build_on_all_text", binary=False)

    # 从附件3的语料中建立word2vec模型
    line_sents = fetch_data("example_3", stop_words=stop, mode="lines")
    wv_model = Word2Vec(line_sents, size=400, window=5, sg=1, min_count=5)

    # 计算所有文档（词列表）的文档向量，对特定词性的词加权,特定词性的词的权重设置为2（其他词默认权重为1）
    weight = {key: 1 for key in set([word for words in specific_only for word in words])}
    docs_vec = [doc_vec_with_weight(doc, model=wv_model, weight=weight)
                for doc in fetch_data("example_3", cut_all=False, mode="lines", stop_words=stop)]

    # 使用pandas进行数据标准化（z-score）
    data = pd.DataFrame(docs_vec)
    data_zs = (data - data.mean()) / data.std()

    # 均值漂移（Mean Shift）模型
    ms_model = MeanShift()
    ms_model.fit(data_zs)  # 拟合模型

    # Kmeans模型
    # km_model = KMeans(n_clusters=5, max_iter=500)
    # km_model.fit(data_zs)

    labels = ms_model.labels_
    output_cluster(labels, lines)


if __name__ == '__main__':
    main()
