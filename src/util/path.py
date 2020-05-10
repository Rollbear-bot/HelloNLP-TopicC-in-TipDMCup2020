# -*- coding: utf-8 -*-
# @Time: 2020/5/8 0:47
# @Author: Rollbear
# @Filename: path.py
# 路径配置文件

sheet_2_input = "../resources/full_dataset/full_dataset_sheet_2.xlsx"  # 附件二的存储路径
sheet_3_input = "../resources/full_dataset/full_dataset_sheet_3.xlsx"  # 附件三的存储路径
sheet_4_input = "../resources/full_dataset/full_dataset_sheet_4.xlsx"  # 附件四的存储路径
sheet_4_labeled_input = "../resources/full_dataset/sheet_4_labeled.xlsx"  # 带人工标注的附件四
stop_words_input = "../resources/special-words/stop_words.txt"  # 停用词表的存储路径

cluster_sheet_path = "./top5_cluster_sheet.xlsx"  # 热点问题表的存储路径（输出路径）
detail_sheet_path = "./detail_sheet.xlsx"  # 留言明细表的存储路径（输出路径）

line_sentence_output = "../resources/temp/full_dataset_cut_all.txt"  # 语料库缓存
word2vec_model_path = "../resources/wv_model/wv_model_0507"  # Word2Vec模型的存储路径
knn_model_path = "../resources/knn_model/knn_model_0507"  # kNN分类器的存储路径
knn_model_target_names = "../resources/knn_model/target_names_0507.txt"  # kNN分类器对应的标签名称
integrity_clf_path = "../resources/label_propagation_integrity_clf_0507"  # 完整性指标分类器路径
interpretability_clf_path = "../resources/label_spreading_interpretability_clf_0507"  # 可解释性指标分类器路径
