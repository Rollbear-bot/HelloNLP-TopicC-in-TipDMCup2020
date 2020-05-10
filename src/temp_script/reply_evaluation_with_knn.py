# -*- coding: utf-8 -*-
# @Time: 2020/5/6 12:49
# @Author: Rollbear
# @Filename: reply_evaluation_with_knn.py

import sklearn
from gensim.models import Word2Vec
from sklearn import metrics  # 模型评价工具
from sklearn.neighbors import KNeighborsClassifier  # 最近邻分类器

from util.dataset import *
from util.vec import doc_vec


def main():
    stop = fetch_default_stop_words()
    comm = fetch_data(ds_name="sheet_4_labeled", cut_all=False, stop_words=stop)

    # 已标注的100个数据
    labeled = {key: c for key, c in comm.items() if c.integrity != None}
    # test
    label_1 = len([record for record in labeled.values() if record.integrity == 1])
    label_2 = len([record for record in labeled.values() if record.interpretability == 1])

    # 建立基于答复文本的Word2Vec模型
    line_sents = fetch_data("sheet_4_labeled", cut_all=False, stop_words=stop,
                            mode="reply_lines", remove_duplicates=False)
    wv_model = Word2Vec(line_sents,
                        size=400, window=5, sg=1, min_count=5)
    wv_model.wv.save_word2vec_format("../resources/wv_reply_text", binary=False)  # 保存模型
    print("model is saved.")

    # 从已标注的数据中分出一部分作为测试集
    xy_labeled = [(doc_vec(c.seg_reply, model=wv_model), c.interpretability) for c in labeled.values()]
    x_labeled = [t[0] for t in xy_labeled]
    y_labeled = [t[1] for t in xy_labeled]
    # 从100个标注样本中分出30个作为测试集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_labeled, y_labeled, test_size=0.3)

    # 训练标记传播模型
    clf = KNeighborsClassifier()  # knn模型
    clf.fit(x_train, y_train)

    predicted = clf.predict(x_test)

    print(metrics.classification_report(y_test, predicted))

    # 0.63-0.8


if __name__ == '__main__':
    main()