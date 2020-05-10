# -*- coding: utf-8 -*-
# @Time: 2020/4/16 22:07
# @Author: Rollbear
# @Filename: naive_bayes.py

# sklearn自带的向量化工具
from sklearn.feature_extraction.text import CountVectorizer
# sklearn自带的TF-TDF构造器
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# 朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB

from entity.comm import Comm
from util.txt_read import load_word_list
from util.xl import read_xl_by_line


def main():
    # 构造向量化工具
    count_vect = CountVectorizer()
    comments = read_xl_by_line("../../resources/xls/e2.xlsx")  # 留言文本
    stop_words = load_word_list("../../resources/special-words/stop_words.txt")  # 停用词
    comm_dict_2 = Comm.generate_comm_dict(comments, True, stop_words)

    line_sents = [comm_dict_2[row[0]].seg_topic + comm_dict_2[row[0]].seg_detail for row in comments]
    sents = list(map(lambda x: " ".join(x), line_sents))

    # 表2中涉及的所有一级标签
    target_names = list(set([row[5] for row in comments]))
    # 将表2中标注的所有一级标签转化成数字表示（target_names中的index）
    targets = [target_names.index(row[5]) for row in comments]

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test \
        = train_test_split(sents, targets, test_size=0.3)

    x_train_counts = count_vect.fit_transform(x_train)  # 拟合模型

    # 构建TF-TDF特征
    tf_transformer = TfidfTransformer().fit(x_train_counts)
    # 构建TF-IDF特征
    x_train_tf = tf_transformer.transform(x_train_counts)

    # 使用特征集（X）和目标（target）训练/拟合一个朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(x_train_tf.toarray(), y_train)

    test_tf = tf_transformer.transform(count_vect.transform(x_test))

    predicted = clf.predict(test_tf.toarray())  # 分类器

    print(classification_report(y_true=y_test, y_pred=predicted, target_names=target_names))
    # print(predicted - y_test)


if __name__ == '__main__':
    main()
