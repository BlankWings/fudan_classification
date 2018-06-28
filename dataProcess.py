# 数据预处理函数，实现对原始数据的分词，去停用词，格式转换等功能
from helper import *
import re
import jieba
import time
import multiprocessing
import chardet

def process(rawFilePath, processDir):# rawFilePath类似于../data/rawData/train/C11-Space
    label = str(rawFilePath).split("/")[-1]
    contents = [] # 储存每一类型的所有文件，每个文件预处理后作为contents的一个元素
    subfile = os.listdir(rawFilePath)
    for file in subfile:
        sentence = ""  # 储存可识别utf-8编码的句子
        result = "" # 储存预处理后的文章内容
        fileName = os.path.join(rawFilePath, file)
        with open(fileName, "rb") as f:
            content = f.read()  # 二进制句子
        for word in jieba.cut(content):
            sentence += word    # 通过jieba分词， 转换为储存可识别utf-8编码的句子， 多了一遍分词使效率降低了
        sentence = re.sub(r'[^\u4e00-\u9fa5]', "",sentence)
        for word in jieba.cut(sentence):
            if word not in STOPWORD_LIST:
                result += word + " "
        contents.append(result.strip())
    processFile = os.path.join(processDir, label+".txt") # 生成C11-Space.txt文件
    with open(processFile, "w", encoding="utf-8") as f:
        for content in contents:
            f.writelines(content + "\n")
    print("{}，文件正在进行预处理！！".format(rawFilePath))

if __name__=="__main__":
    st =time.time()
    print("开始进行文件与处理！")
    subfolder = os.listdir(RAW_TRAIN_DATA_PATH)
    pool = multiprocessing.Pool(processes=12)
    for folder in subfolder:
        folderName = os.path.join(RAW_TRAIN_DATA_PATH, folder)
        pool.apply_async(process, (folderName, PROCESS_TRAIN_DATA_PATH))
    pool.close()
    pool.join()
    subfolder = os.listdir(RAW_TEST_DATA_PATH)
    pool = multiprocessing.Pool(processes=12)
    for folder in subfolder:
        folderName = os.path.join(RAW_TEST_DATA_PATH, folder)
        pool.apply_async(process, (folderName, PROCESS_TEST_DATA_PATH))
    pool.close()
    pool.join()
    et =time.time()
    print("总计用时：{:.3f}s！！".format(et-st))


