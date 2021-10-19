"""
手写数字识别
2021-10-19
数据集包括两部分，一部分是训练数据集，共有1934个数据；
另一部分是测试数据集，共有946个数据。
所有数据命名格式都是统一的，例如数字5的第56个样本——5_56.txt
算法步骤


收集数据：公开数据源
分析数据，构思如何处理数据
导入训练数据，转化为结构化的数据格式
计算距离（欧式距离）
导入测试数据，计算模型准确率
手写数字，实际应用模型
"""

# 处理文本文件
import operator
from os import listdir
import numpy as np

from PIL import Image


def img_to_vector(filename):
    the_matrix = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            the_matrix[0, 32 * i + j] = int(line_str[j])  # 将32*32=1024个元素赋值给一维零矩阵
    return the_matrix


def classify0(test_date, train_data, label, k):
    size = train_data.shape[0]
    # 将测试数据每一行复制Size次减去训练数据，横向复制Size次，纵向复制1次
    the_matrix = np.tile(test_date, (size, 1)) - train_data
    sq_the_matrix = the_matrix ** 2
    # 平方加和，axis = 1 代表横向
    all_the_matrix = sq_the_matrix.sum(axis=1)
    distance = all_the_matrix ** 0.5
    sort_distance = distance.argsort()
    dis_dict = {}
    for i in range(k):
        the_label = label[sort_distance[i]]
        # 将标签的key和value传入字典
        dis_dict[the_label] = dis_dict.get(the_label, 0) + 1
    # 将字典按value值的大小排序，由大到小，即在K范围内，筛选出现次数最多几个标签
    sort_count = sorted(dis_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sort_count[0][0]


def handWritingClassTeat():
    labels = []
    train_data = listdir("trainingDigits")
    m_train = len(train_data)
    # 生成一个列数为train_matrix，行为1024的零矩阵
    train_matrix = np.zeros((m_train, 1024))
    for i in range(m_train):
        file_name_str = train_data[i]  # 获取某一个文件名
        file_str = file_name_str.split('.')[0]  # 去掉.txt
        # 获取第一个字符，即它是哪个数字
        file_num = int(file_str.split('_')[0])
        labels.append(file_num)
        train_matrix[i, :] = img_to_vector("trainingDigits/%s" % file_name_str)
    error = []
    test_matrix = listdir("testDigits")
    correct = 0.0
    m_test = len(test_matrix)
    for i in range(m_test):
        file_name_str = test_matrix[i]
        file_str = file_name_str.split('.')[0]
        file_num = int(file_str.split('_')[0])
        test_classify = img_to_vector("testDigits/%s" % file_name_str)
        classify_result = classify0(test_classify, train_matrix, labels, 4)
        print("the consequence of protest:%s\tTrue consequence:%s" % (classify_result, file_num))
        if classify_result == file_num:
            correct += 1.0
        else:
            error.append((file_name_str, classify_result))
    print("the correct rate:{:.2f}%".format(correct / float(m_test) * 100))
    # print(error)
    print(len(error))


"""
训练数据和测试数据都是txt格式的，
而现在却都是自己手写的数字并拍出来的照片，
因此我需要将照片转成32×32的txt文本
首先，拍出来的图片都是彩色图片，它的内部格式是3维的，
而我们要让其为二维的，首先得将它转换为灰度图片
"""


def trans_pic_to_grey(origin_path):
    ima = Image.open(origin_path)
    ima = ima.resize((32, 32))
    grey_image = ima.convert("L")
    save_path = origin_path.split('.')[-2] + "_grey.jpg"
    grey_image.save(save_path)
    return save_path


# 32×32的灰度图片，将其变成0和1
def trans_grey_to_binary(origin_path):
    imag = Image.open(origin_path)
    save_path = origin_path.split('.')[-2] + "_.txt"
    fh = open(save_path, 'w')
    for i in range(imag.size[1]):
        for j in range(imag.size[0]):
            color = imag.getpixel((j, i))
            if color > 100:
                color = 0
            else:
                color = 1
            fh.write(str(color))
        fh.write("\n")
    fh.close()
    return save_path


def my_handWriting_image_distinguish(pic_path):
    pic_path = trans_pic_to_grey(pic_path)
    pic_path = trans_grey_to_binary(pic_path)
    hw_labels = []
    train_data = listdir("trainingDigits")
    m_train = len(train_data)
    # 生成一个列数为train_matrix，行为1024的零矩阵
    train_matrix = np.zeros((m_train, 1024))
    for i in range(m_train):
        file_name_str = train_data[i]  # 获取某一个文件名
        file_str = file_name_str.split('.')[0]  # 去掉.txt
        # 获取第一个字符，即它是哪个数字
        file_num = int(file_str.split('_')[0])
        hw_labels.append(file_num)
        train_matrix[i, :] = img_to_vector("trainingDigits/%s" % file_name_str)
    vector_under_test = img_to_vector(pic_path)
    classify_result = classify0(vector_under_test, train_matrix, hw_labels, 5)
    print("result is : %d" % int(classify_result))


if __name__ == "__main__":
    # handWritingClassTeat()
    for i in range(3):
        my_handWriting_image_distinguish("image%d"%i + '.jpg')
