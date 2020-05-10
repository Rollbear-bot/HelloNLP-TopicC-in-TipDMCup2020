# -*- coding: utf-8 -*-
# @Time: 2020/5/8 1:07
# @Author: Rollbear
# @Filename: issue3.py
# 问题三：答复信息评价器

import gensim

from util.dataset import *
from util.evaluation import ReplyEvaluation
from util.path import *
from util.standardization import ScoreStandardization


def main():
    stop = fetch_default_stop_words()
    sheet_4 = fetch_data("full_dataset_sheet_4", stop_words=stop, remove_duplicates=False)

    # 加载word2vec模型
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=False)

    # 初始化评价器：设置评价模型的路径
    ReplyEvaluation.load_integrity_clf(integrity_clf_path)
    ReplyEvaluation.load_interpretability_clf(interpretability_clf_path)

    for key, comm in sheet_4.items():
        # 为表中的留言对象动态绑定一个“答复分数”属性
        sheet_4[key].reply_score = ReplyEvaluation(comm, wv_model).score

    # 所有答复文本的得分平均值（即构造常模）
    # mean = sum([comm.reply_score for comm in sheet_4.values()]) / len(sheet_4)
    # variance = sum([(comm.reply_score - mean)**2 for comm in sheet_4.values()]) / len(sheet_4)

    # 计算常模
    model = [comm.reply_score for comm in sheet_4.values()]
    ScoreStandardization.set_max_sc(5)
    ScoreStandardization.set_model(model)

    # 常模参照计分
    score_lt = [ScoreStandardization(comm.reply_score).score for comm in sheet_4.values()]

    # 至此，答复信息的评价与计分已完成，分数值按照输入表的顺序存储在score_lt列表中
    print(score_lt)


if __name__ == '__main__':
    main()
