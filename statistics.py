# 统计一些内容
from helper import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 统计训练集和测试集一共多少篇, 还有训练集和测试集每个类别各有多少篇（尝试解决数据不均衡问题）
def counterFile(rawDataPath): # rawDataPath类似于../data/rawData/train
    subFolderList = os.listdir(rawDataPath)  # subFolderList为[...,C11-Space,...]
    cnt = 0; categoryList, numberList= [],[]  # cnt用于统计文章的数目， class_number是统计类别文章数的字典
    for subFolder in subFolderList:    # subFolder为C11-Space之类
        subFolderName = os.path.join(rawDataPath, subFolder) # subFolderName类似于../data/rawData/train/C11-Space
        subFileList = os.listdir(subFolderName)
        number = len(subFileList) # len(subFileList)为文章的数目
        cnt += number
        numberList.append(number)
        category = subFolder.split("C")[1].split("-")[0]
        categoryList.append(category)
    count = "{}文件夹共有语料：{}篇。".format(rawDataPath, cnt)
    return count, categoryList, numberList

# 统计预处理后的本文的基本数据，包括文本最短长度，最大长度等
def describeContent(data_path):  # data_path例如../processData/train
    words_list = []
    subfile_list = os.listdir(data_path)   # subfile_list为[...,C11-Space.txt,...]
    for subfile in subfile_list: # subfile为C11-Space.txt之类
        subfilepath = os.path.join(data_path, subfile)  # 获取文件路径
        with open(subfilepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words_lenths = len(line.split(" "))
                words_list.append(words_lenths)
                if words_lenths < 10:
                    print(line)
                    print(subfile)
    words_series = pd.Series(words_list)
    return words_series

if __name__ == '__main__':
    train_count, train_categoryList, train_numberList  = counterFile(RAW_TRAIN_DATA_PATH)
    test_count, test_categoryList, test_numberList = counterFile(RAW_TEST_DATA_PATH)
    # 画出训练集和测试集各个类别的文章分布情况
    # plt.bar(train_categoryList, train_numberList)
    # plt.xlabel("category")
    # plt.ylabel("number")
    # plt.title("train_set")
    # for x,y  in zip(train_categoryList, train_numberList):
    #     plt.text(x,y,y,ha="center",va="bottom")
    # plt.show()
    # plt.bar(test_categoryList, test_numberList, color="r")
    # plt.xlabel("category")
    # plt.ylabel("number")
    # plt.title("test_set")
    # for x, y in zip(test_categoryList, test_numberList):
    #     plt.text(x, y, y, ha="center", va="bottom")
    # plt.show()
    # 统计所有篇章的单词长度信息
    train_words = describeContent(PROCESS_TRAIN_DATA_PATH)
    plt.bar(train_words.index, train_words)
    plt.show()
    print(train_words.describe())

