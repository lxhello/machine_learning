import numpy as np
import operator


def create_data_set():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ["love", "love", "fight", "fight"]
    return group, labels


def classify(inX, dataSet, labels, k):
    data_set_size = dataSet.shape[0]  # numpy函数shape[0]返回dataSet的行数
    diff_mat = np.tile(inX, (data_set_size, 1)) - dataSet  # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    sq_diff_mat = diff_mat ** 2  # 二维特征相减后平方
    sq_distance = sq_diff_mat.sum(axis=1)  # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    distance = sq_distance ** 0.5
    sort_dist_indices = distance.argsort()  # 返回distances中元素从小到大排序后的索引值
    class_count = {}
    for i in range(k):
        # 取出前k个元素的类别
        vote_label = labels[sort_dist_indices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # reverse降序排序字典,返回次数最多的类别
    return sort_class_count[0][0]


if __name__ == "__main__":
    group, labels = create_data_set()
    test = [102, 51]
    test_class = classify(test, group, labels, 2)
    print(test_class)
