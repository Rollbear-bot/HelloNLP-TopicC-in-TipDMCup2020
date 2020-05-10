class ScoreStandardization:
    """使用常模参照计分法进行分数转化"""
    max_score = None
    mean = None
    variance = None
    model = []  # 常模

    @classmethod
    def set_max_sc(cls, sc):
        """设置最大分值"""
        ScoreStandardization.max_score = sc

    @classmethod
    def set_model(cls, model_lt: list):
        """设置常模"""
        ScoreStandardization.model = sorted(model_lt)

    # def __init__(self):
    #     score_list = [100, 70, 65, 55, 50, 45, 40]
    #     total_score = 0
    #     mid_variance = 0
    #
    #     μ = 500  # 正态分布可变参数
    #     σ = 70  # 正态分布可变参数
    #
    #     for i in range(len(score_list)):
    #         total_score = total_score + score_list[i]
    #     print(total_score)  # 总分
    #
    #     avg_score = round(total_score / len(score_list), 2)
    #     print(avg_score)  # 平均分
    #
    #     for i in range(len(score_list)):
    #         mid_variance = mid_variance + (score_list[i] - avg_score) ** 2
    #     variance = round(mid_variance / len(score_list), 2)
    #     print(variance)  # 方差
    #
    #     standard_deviation = round(math.sqrt(variance), 2)
    #     print(standard_deviation)  # 标准差
    #
    #     final_score_list = []
    #     for i in range(len(score_list)):
    #         final_score = round((score_list[i] - avg_score) / standard_deviation * σ + μ, 2)
    #         final_score_list.append(final_score)
    #
    #     print(final_score_list)  # 最终得分

    def __init__(self, sample):
        """
        :param sample: 样本，即分数值
        """
        index = 0
        for index in range(len(ScoreStandardization.model)-1):
            if ScoreStandardization.model[index] <= sample < ScoreStandardization.model[index + 1]:
                break
        self.rank = index / len(ScoreStandardization.model)

    @property
    def score(self):
        return round(ScoreStandardization.max_score * self.rank, 2)
