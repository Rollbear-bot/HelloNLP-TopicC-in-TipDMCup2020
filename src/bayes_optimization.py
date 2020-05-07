# -*- coding: utf-8 -*-
# @Time: 2020/5/6 23:52
# @Author: Rollbear
# @Filename: bayes_optimization.py
# 使用贝叶斯优化方法优化kNN分类模型的超参数

import sklearn
from bayes_opt import BayesianOptimization  # 贝叶斯优化器
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from util.dataset import *
from util.vec import doc_vec

# 全局变量
stop = fetch_default_stop_words()  # 停用词表
# 附件二、三、四中“留言主题”和“留言详情”属性的全部语料(cut_all=False)
# line_sents = LineSentence("../resources/temp/full_dataset_text.txt")

# 读取表2
comments = read_xl_by_line("../resources/full_dataset/full_dataset_sheet_2.xlsx")
comm_dict_2 = fetch_data("full_dataset_sheet_2",
                         stop_words=stop, cut_all=True, remove_duplicates=False)
comm_dict_3 = fetch_data("full_dataset_sheet_3",
                         stop_words=stop, cut_all=True, remove_duplicates=False)
comm_dict_4 = fetch_data("full_dataset_sheet_4",
                         stop_words=stop, cut_all=True, remove_duplicates=False)

# 将三个表分好词的语料输出到文本文件
text_file = open("../resources/temp/full_dataset_cut_all.txt", "w", encoding="utf8")
for d in (comm_dict_2, comm_dict_3, comm_dict_4):
    for c in d.values():
        text_file.write(" ".join(c.seg_topic) + "\n")  # 留言主题
        text_file.write(" ".join(c.seg_detail) + "\n")  # 留言详情
text_file.close()
# 附件二、三、四中“留言主题”和“留言详情”属性的全部语料(cut_all=True)
line_sents = LineSentence("../resources/temp/full_dataset_text.txt")

# 表2中涉及的所有一级标签
target_names = list(set([row[5] for row in comments]))
# 将表2中标注的所有一级标签转化成数字表示（target_names中的index）
targets = [target_names.index(row[5]) for row in comments]


def evaluate(**bo_params):
    """
    模型评估函数
    :param bo_params: BO自带的参数
    :return: 评估值，BO会朝着评估值高的方向优化
    """
    # 固定的超参数（无需调参的）
    param = {}
    # BO生成的参数
    # param.update(bo_params)

    # 贝叶斯优化器生成的超参数
    param["wv_size"] = int(bo_params['wv_size'])
    param["wv_window"] = int(bo_params["wv_window"])
    param["wv_min_count"] = int(bo_params["wv_min_count"])
    param["knn_leaf_size"] = int(bo_params["knn_leaf_size"])
    param["knn_n_neighbors"] = int(bo_params["knn_n_neighbors"])
    param["alpha"] = float(bo_params["alpha"])

    word2vec_model = \
        Word2Vec(line_sents,
                 size=param['wv_size'],
                 window=param['wv_window'],
                 sg=1,
                 min_count=param['wv_min_count'],
                 alpha=param["alpha"])
    # word2vec_model.wv.save_word2vec_format("../resources/word2vec_model", binary=False)

    # 计算所有评论的文档向量
    comms_vec = [doc_vec(
        comm_dict_2[row[0]].seg_topic + comm_dict_2[row[0]].seg_detail,
        word2vec_model)
        for row in comments]

    # 交叉验证（5轮）
    acc = []
    for count in range(5):
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(comms_vec, targets, test_size=0.3)

        # 引入KNN模型
        knn = KNeighborsClassifier(leaf_size=param['knn_leaf_size'],
                                   n_neighbors=param['knn_n_neighbors'])
        # 训练knn模型
        knn.fit(X_train, y_train)
        predicted = knn.predict(X_test)
        # 计算模型精度（F1）
        accuracy = metrics.accuracy_score(y_test, predicted)
        acc.append(accuracy)

    return sum(acc) / len(acc)  # 5次交叉验证的精确值的平均作为评估值返回


def bayesian_search(clf, params):
    """贝叶斯搜索器"""
    num_iter = 25  # 迭代次数
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    # params = bayes.res['max']
    params = sorted([test for test in bayes.res], key=lambda x: x['target'])[0]['params']
    print(params)
    with open("../resources/log/bo_log_0507.txt", "a") as f:
        f.write(str(params))

    return params


def main():
    params = {"wv_size": (50, 600),
              "wv_window": (6, 40),
              "wv_min_count": (3, 40),
              "knn_leaf_size": (10, 50),
              "knn_n_neighbors": (3, 21),
              "alpha": (0.001, 0.1)}
    bayesian_search(evaluate, params)


if __name__ == '__main__':
    main()
