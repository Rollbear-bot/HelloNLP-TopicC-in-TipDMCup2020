# -*- coding: utf-8 -*-
# @Time: 2020/5/7 23:57
# @Author: Rollbear
# @Filename: train_label_spreading.py
# 训练标签传播模型的脚本

import gensim
from gensim.models import Word2Vec
from sklearn.externals import joblib  # 模型保存与加载
from sklearn.semi_supervised import LabelPropagation  # 标记传播模型
from sklearn.semi_supervised.label_propagation import LabelSpreading

from util.dataset import *
from util.path import *
from util.vec import doc_vec


def main():
    stop = fetch_default_stop_words()
    comm = fetch_data(ds_name="sheet_4_labeled", cut_all=False, stop_words=stop)

    # 将无标注的数据的标记置为-1
    for key in comm.keys():
        if comm[key].integrity is None:
            comm[key].integrity = -1
            comm[key].interpretability = -1

    labeled = {key: c for key, c in comm.items() if c.integrity != -1}
    unlabeled = {key: c for key, c in comm.items() if c.integrity == -1}

    # 加载基于答复文本的wv模型
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False)

    # -------------------------------------------------------
    # 模型一
    xy = [(doc_vec(c.seg_reply, model=wv_model), c.integrity) for c in unlabeled.values()]
    x = [t[0] for t in xy]
    y = [t[1] for t in xy]

    xy_labeled = [(doc_vec(c.seg_reply, model=wv_model), c.integrity) for c in labeled.values()]
    x_labeled = [t[0] for t in xy_labeled]
    y_labeled = [t[1] for t in xy_labeled]

    x_train, y_train = (x_labeled, y_labeled)

    # 将已标注的数据与未标注的数据混合成为训练集
    x_train += x
    y_train += y

    # 训练标记传播模型
    clf = LabelPropagation(gamma=30)  # 模型1
    clf.fit(x_train, y_train)
    joblib.dump(clf, integrity_clf_path)

    # --------------------------------------------------------------
    # 模型二
    xy = [(doc_vec(c.seg_reply, model=wv_model), c.interpretability) for c in unlabeled.values()]
    x = [t[0] for t in xy]
    y = [t[1] for t in xy]

    xy_labeled = [(doc_vec(c.seg_reply, model=wv_model), c.interpretability) for c in labeled.values()]
    x_labeled = [t[0] for t in xy_labeled]
    y_labeled = [t[1] for t in xy_labeled]

    x_train, y_train = (x_labeled, y_labeled)

    # 将已标注的数据与未标注的数据混合成为训练集
    x_train += x
    y_train += y

    # 训练标记传播模型
    clf = LabelSpreading()  # 模型2
    clf.fit(x_train, y_train)
    joblib.dump(clf, interpretability_clf_path)


if __name__ == '__main__':
    main()
