# -*- coding: utf-8 -*-
# @Time: 2020/4/19 17:09
# @Author: Rollbear
# @Filename: topic_modeling.py
# 主题建模

from gensim import corpora  # 语料/词频工具
from gensim.models.ldamodel import LdaModel  # LDA模型

from util.dataset import fetch_data, fetch_default_stop_words


def main():
    stop = fetch_default_stop_words()
    stop.extend(["\t", "", "\n"])  # 这里另外添加几个停用词

    # 分好词的行文本，使用精准模式
    line_sents = fetch_data(ds_name="example_3", mode="lines", cut_all=False, stop_words=stop)

    # 建立doc-term矩阵
    docs = [" ".join(lt) for lt in line_sents]  # 处理成“每个文档是一个字符串，词与词之间用空格隔开”的形式
    dictionary = corpora.Dictionary(line_sents)  # 建立词典
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in line_sents]  # 文档-词频矩阵

    # 训练LDA模型
    lda_model = LdaModel(doc_term_matrix, num_topics=5, id2word=dictionary, passes=50)


if __name__ == '__main__':
    main()

