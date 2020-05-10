# -*- coding: utf-8 -*-
# @Time: 2020/4/14 12:12
# @Author: Rollbear
# @Filename: from_word2vec_to_knn.py
# 如何完成从Word2Vec向量到KNN的输入矩阵的转换？

import time

import sklearn
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn import metrics  # 模型评价工具
from sklearn.neighbors import KNeighborsClassifier  # 最近邻分类器

from util.dataset import *
from util.vec import doc_vec

sheet_2_input = "../resources/full_dataset/full_dataset_sheet_2.xlsx"  # 附件2的输入路径
sheet_3_input = "../resources/full_dataset/full_dataset_sheet_3.xlsx"  # 附件3的输入路径
sheet_4_input = "../resources/full_dataset/full_dataset_sheet_4.xlsx"  # 附件4的输入路径
stop_words_input = "../resources/special-words/stop_words.txt"  # 停用词表输入路径

wv_model_output = "../resources/wv_model/wv_model_full_dataset_0502"  # word2vec模型输出路径
wv_model_tmp_output = "../resources/wv_model/wv_model_sheet_2_only_0502"  # 备用模型输出路径
line_sentence_output = "../resources/temp/full_dataset_text.txt"  # 行文本输出路径
sheet_2_line_sentence_output = "../resources/temp/sheet_2_text.txt"  # 附件表2行文本输出路径
log_output = "../resources/log/test_wv_size_log_0503.txt"  # 日志输出路径


def main():
    # -----------------调参窗口------------------
    # 列出要尝试的所有参数取值
    wv_size_lt = [550]
    wv_window_lt = [18]
    wv_sg_lt = [1]
    wv_min_count_lt = [8]
    knn_leaf_size_lt = [30]
    knn_n_neighbors_lt = [12]
    cut_all = False

    # 填充参数
    params = []
    keys = ['wv_size', 'wv_window', 'wv_sg', 'wv_min_count', 'knn_leaf_size', 'knn_n_neighbors']
    for wv_size in wv_size_lt:
        for wv_window in wv_window_lt:
            for wv_sg in wv_sg_lt:
                for wv_min_count in wv_min_count_lt:
                    for knn_leaf_size in knn_leaf_size_lt:
                        for knn_n_neighbors in knn_n_neighbors_lt:
                            param = {keys[0]: wv_size,
                                     keys[1]: wv_window,
                                     keys[2]: wv_sg,
                                     keys[3]: wv_min_count,
                                     keys[4]: knn_leaf_size,
                                     keys[5]: knn_n_neighbors}
                            params.append(param)
    # ------------------------------------------

    # 读取表2
    comments = read_xl_by_line(sheet_2_input)
    # 读取附件三、四到元组
    sheet_3 = read_xl_by_line(sheet_3_input)
    sheet_4 = read_xl_by_line(sheet_4_input)

    # 分词、去停用词并生成表2的评论对象字典
    stop_words = load_word_list(stop_words_input)
    comm_dict_2 = Comm.generate_comm_dict(comments, cut_all=cut_all, stop_words_lt=stop_words)
    comm_dict_3 = Comm.generate_comm_dict(sheet_3, cut_all=cut_all, stop_words_lt=stop_words)
    comm_dict_4 = Comm.generate_comm_dict(sheet_4, cut_all=cut_all, stop_words_lt=stop_words)

    # 将三个表分好词的语料输出到文本文件
    text_file = open(line_sentence_output, "w", encoding="utf8")
    for d in (comm_dict_2, comm_dict_3, comm_dict_4):
        for c in d.values():
            text_file.write(" ".join(c.seg_topic) + "\n")  # 留言主题
            text_file.write(" ".join(c.seg_detail) + "\n")  # 留言详情
    text_file.close()

    # 表2中涉及的所有一级标签
    target_names = list(set([row[5] for row in comments]))
    # 将表2中标注的所有一级标签转化成数字表示（target_names中的index）
    targets = [target_names.index(row[5]) for row in comments]

    logs = []  # 日志记录

    for index, param in enumerate(params):
        # 测试每一种参数组合下的效果

        # 训练word embedding模型（从表2、3、4的混合语料构建）
        word2vec_build_on_all_text = \
            Word2Vec(LineSentence(line_sentence_output),
                     size=param['wv_size'],
                     window=param['wv_window'],
                     sg=param['wv_sg'],
                     min_count=param['wv_min_count'])
        # word2vec_build_on_all_text.wv.save_word2vec_format("../resources/word2vec_build_on_all_text", binary=False)

        # 计算所有评论的文档向量
        comms_vec = [doc_vec(
                        comm_dict_2[row[0]].seg_topic + comm_dict_2[row[0]].seg_detail,
                        word2vec_build_on_all_text)
                     for row in comments]

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(comms_vec, targets, test_size=0.3)

        # 引入KNN模型
        knn = KNeighborsClassifier(leaf_size=param['knn_leaf_size'],
                                   n_neighbors=param['knn_n_neighbors'])

        # 训练knn模型
        knn.fit(X_train, y_train)
        predicted = knn.predict(X_test)

        # 生成测试日志
        accuracy = metrics.accuracy_score(y_test, predicted)
        log = f"Sample({index+1}/{len(params)})\n" \
              + "Param: " + str(param) + "\n" \
              + f"Accuracy: {accuracy}\n"
        logs.append([log, accuracy, str(param)])

        # 输出完整报告
        print(metrics.classification_report(y_test, predicted, target_names=target_names))

        # 将测试结果输出到控制台
        print(log)

    # 所有参数组合的测试结束后，输出日志到文本文件
    file = open(log_output, 'w', encoding='utf8')
    file.writelines([log[0] for log in logs])

    # 计算所测试的参数组合中的最优解，生成测试报告
    logs_sorted = sorted(logs, key=lambda x: -x[1])  # 按准确率排序
    report = "\n[REPORT] Top 10:\n"
    for log in logs_sorted[:10]:
        report += f"Accuracy: {log[1]}\nParam: {log[2]}\n"
    report += str(time.asctime(time.localtime(time.time())))  # 结束时打上时间戳
    file.write(report)
    file.close()


if __name__ == '__main__':
    main()
