# -*- coding: utf-8 -*-
# @Time: 2020/5/7 23:13
# @Author: Rollbear
# @Filename: train_wv.py
# 0.01667  |  11.69    |  13.58    |  6.451    |  176.3    |  36.91

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from entity.comm import Comm
from util.txt_read import load_word_list
from util.xl import read_xl_by_line

sheet_2_input = "../resources/full_dataset/full_dataset_sheet_2.xlsx"
sheet_3_input = "../resources/full_dataset/full_dataset_sheet_3.xlsx"
sheet_4_input = "../resources/full_dataset/full_dataset_sheet_4.xlsx"
stop_words_input = "../resources/special-words/stop_words.txt"
cut_all = True
line_sentence_output = "../resources/temp/full_dataset_cut_all.txt"
word2vec_model_path = "../resources/wv_model/wv_model_0507"


def main():
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

    # 训练word2vec模型（使用贝叶斯优化得出的一组参数）
    word2vec_model = Word2Vec(LineSentence(line_sentence_output),
                              alpha=0.01667,
                              min_count=6,
                              size=176,
                              window=36)
    # 文本模式保存word2vec模型
    word2vec_model.wv.save_word2vec_format(word2vec_model_path, binary=False)


if __name__ == '__main__':
    main()
