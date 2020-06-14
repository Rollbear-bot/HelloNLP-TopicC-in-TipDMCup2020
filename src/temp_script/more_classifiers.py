# -*- coding: utf-8 -*-
# @Time: 2020/6/4 15:58
# @Author: Rollbear
# @Filename: more_classifiers.py
# 评估线性SVM在解决留言分类问题中的可行性

import sklearn
from sklearn import metrics  # 模型评价工具
# sklearn自带的向量化工具
from sklearn.feature_extraction.text import CountVectorizer
# sklearn自带的TF-TDF构造器
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # 最近邻分类器
from sklearn.svm import SVC  # 支持向量机分类器

from util import dataset
from util.path import *
from util.xl import read_xl_by_line


# 朴素贝叶斯分类器


def svm_classifier():
    vec, targets, target_names = dataset.fetch_issue1_dataset()

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test \
        = sklearn.model_selection.train_test_split(vec, targets, test_size=0.3)

    model = SVC(kernel='linear')  # 线性核SVM
    model.fit(x_train, y_train)  # 训练SVM模型

    predicted = model.predict(x_test)  # 使用SVM预测

    # 模型评价
    accuracy = metrics.accuracy_score(y_test, predicted)
    print(accuracy)

    # 精度试验记录
    # 0.827723488961274
    # 0.8306188925081434
    # 0.8280854144046327
    # 0.832066594281578


def naive_bayes_cls():
    pass  # 使用kaggle上的记录
    # vec, targets, target_names = dataset.fetch_issue1_dataset()
    # # 分割训练集和测试集
    # x_train, x_test, y_train, y_test \
    #     = sklearn.model_selection.train_test_split(vec, targets, test_size=0.3)
    #
    # model = MultinomialNB()
    # model.fit(x_train, y_train)
    # predicted = model.predict(x_test)
    #
    # acc = metrics.accuracy_score(y_test, predicted)
    # print(acc)


def knn_cls_anew():
    vec, targets, target_names = dataset.fetch_issue1_dataset()
    x_train, x_test, y_train, y_test \
        = sklearn.model_selection.train_test_split(vec, targets, test_size=0.3)

    knn = KNeighborsClassifier(leaf_size=11,
                               n_neighbors=13)
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test)

    acc = metrics.accuracy_score(y_test, predicted)
    print(acc)

    # 0.8089033659066233
    # 0.8060079623597539


def tf_idf_knn_clf():
    """使用tf-idf作为特征的kNN分类器"""
    sheet_2 = read_xl_by_line(sheet_2_input)
    stop = dataset.fetch_default_stop_words()
    comm_dict_2 = dataset.fetch_data("full_dataset_sheet_2", stop_words=stop,
                                     cut_all=False, remove_duplicates=False)

    _, targets, target_names = dataset.fetch_issue1_dataset()

    seg_sheet_2 = [comm_dict_2[row[0]].seg_topic + comm_dict_2[row[0]].seg_detail for row in sheet_2]
    sents = list(map(lambda x: " ".join(x), seg_sheet_2))

    count_vect = CountVectorizer()

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test \
        = train_test_split(sents, targets, test_size=0.3)

    x_train_counts = count_vect.fit_transform(x_train)  # 拟合模型

    # 构建TF-TDF特征
    tf_transformer = TfidfTransformer().fit(x_train_counts)
    # 构建TF-IDF特征
    x_train_tf = tf_transformer.transform(x_train_counts)

    # 使用特征集（X）和目标（target）训练/拟合一个kNN分类器
    clf = KNeighborsClassifier(algorithm="brute", leaf_size=11, n_neighbors=13)
    clf.fit(x_train_tf.toarray(), y_train)

    # 将x_test中的文本转化为tf-idf矩阵表示
    test_tf = tf_transformer.transform(count_vect.transform(x_test))
    predicted = clf.predict(test_tf)  # 预测x_test中的分类

    acc = metrics.accuracy_score(y_test, predicted)
    # f1 = metrics.f1_score(y_test, predicted)
    print(f"acc: {acc}")
    # print(f"f1-score: {f1}")
    # 实验精度
    # 0.822294607310894, default
    # 0.8479913137893594, leaf_size=11, n_neighbors=13
    # acc: 0.8487151646760768, leaf_size=11, n_neighbors=13


if __name__ == '__main__':
    # svm_classifier()
    # naive_bayes_cls()
    # knn_cls_anew()
    tf_idf_knn_clf()
