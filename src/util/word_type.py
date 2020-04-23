# -*- coding: utf-8 -*-
# @Time: 2020/4/23 11:19
# @Author: Rollbear
# @Filename: word_type.py
# 词性工具

import jieba
import jieba.posseg as pseg  # 使用jieba模块做词性标注


def pick_specific_type_words(text: list, types: list, paddle=True, stop_words=None):
    """
    挑选出指定词性的词
    :param text: 文档的列表，每一个文档（留言文本）为一个字符串
    :param types: 指定词性标签组成的列表
    :param paddle: 是否打开paddle
    :param stop_words: 停用词表，一般来说不需要加载停用词（因为已经指定了特定的词性，
    如果结果中存在大量停用词可以附上停用词表
    :return: list of list(specific words in a document)
    """
    if paddle:
        jieba.enable_paddle()  # 打开飞桨模式

    stop_words = [] if stop_words is None else stop_words

    # 找出那些指定词性的（且不在停用词表中的）词
    res = [[pair[0] for pair in [list(p) for p in pseg.cut(line)]
            if pair[1] in types and pair[0] not in stop_words]
           for line in text]
    return res
