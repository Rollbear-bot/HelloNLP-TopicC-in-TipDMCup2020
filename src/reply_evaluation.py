# -*- coding: utf-8 -*-
# @Time: 2020/5/4 11:53
# @Author: Rollbear
# @Filename: reply_evaluation.py
# 答复信息评价（基于部分手工标注）

from gensim.models import Word2Vec
from sklearn.externals import joblib  # 模型保存与加载
from sklearn.semi_supervised import LabelPropagation  # 标记传播模型

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

    # 测试完整性指标的半监督分类
    xy = [(doc_vec(c.seg_reply, model=wv_model), c.integrity) for c in comm.values()]
    x = [t[0] for t in xy]
    y = [t[1] for t in xy]

    cls = LabelPropagation()
    cls.fit(x, y)
    joblib.dump(cls, "../resources/label_propagation_cls")

    x_test = [doc_vec(c.seg_reply, wv_model) for c in unlabeled.values()]
    y_test = [c.integrity for c in unlabeled.values()]
    print("Accuracy:%f" % cls.score(x_test, y_test))


if __name__ == '__main__':
    main()
