# -*- coding: utf-8 -*-
# @Time: 2020/5/4 11:53
# @Author: Rollbear
# @Filename: reply_evaluation.py
# 答复信息评价（基于部分手工标注）

import sklearn
from gensim.models import Word2Vec
from sklearn.externals import joblib  # 模型保存与加载
from sklearn.semi_supervised.label_propagation import LabelSpreading

from util.dataset import *
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

    # 建立基于答复文本的Word2Vec模型
    line_sents = fetch_data("sheet_4_labeled", cut_all=False, stop_words=stop,
                            mode="reply_lines", remove_duplicates=False)
    wv_model = Word2Vec(line_sents,
                        size=400, window=5, sg=1, min_count=5)
    wv_model.wv.save_word2vec_format("../resources/wv_reply_text", binary=False)  # 保存模型
    print("model is saved.")

    # 加载基于答复文本的wv模型
    # wv_model = gensim.models.KeyedVectors.load_word2vec_format("../resources/wv_reply_text", binary=False)
    # print("model is loaded.")

    xy = [(doc_vec(c.seg_reply, model=wv_model), c.interpretability) for c in unlabeled.values()]
    x = [t[0] for t in xy]
    y = [t[1] for t in xy]

    # 从已标注的数据中分出一部分作为测试集
    xy_labeled = [(doc_vec(c.seg_reply, model=wv_model), c.interpretability) for c in labeled.values()]
    x_labeled = [t[0] for t in xy_labeled]
    y_labeled = [t[1] for t in xy_labeled]
    # 从100个标注样本中分出30个作为测试集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_labeled, y_labeled, test_size=0.3)

    # 将已标注的数据与未标注的数据混合成为训练集
    x_train += x
    y_train += y

    # 训练标记传播模型
    # clf = LabelPropagation(gamma=30)  # 模型1
    clf = LabelSpreading()  # 模型2, 第二组参数：max_iter=100, kernel='rbf', gamma=0.1
    clf.fit(x_train, y_train)
    joblib.dump(clf, "../resources/label_spreading_interpretability_clf")

    # x_test = [doc_vec(c.seg_reply, wv_model) for c in labeled.values()]
    # y_test = [c.integrity for c in labeled.values()]
    print(f"Accuracy:{clf.score(x_test, y_test)}")
    # 完整性指标精度：0.95（默认参数下），可解释性指标精度：0.91（默认参数）；0.72~0.75（参数组合2）
    # 结论：完整性指标的分类应该使用模型一，可解释性的分类使用模型二

    # 完整性：0.76-0.86（模型1，gamma=30）
    # 可解释性：0.63-0.75（模型1）
    # 模型2 0.7-0.77


if __name__ == '__main__':
    main()
