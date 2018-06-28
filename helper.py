# 储存文件路径, 基本参数和辅助函数
import os

BASE_DIR = os.path.dirname(os.getcwd()) # 提取当前文件夹的上级文件夹作为基础路径
RAW_TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/rawData/train") # 原始训练文件路径
RAW_TEST_DATA_PATH = os.path.join(BASE_DIR, "data/rawData/test") # 原始测试文件路径
PROCESS_TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/processData/train") # 预处理后的训练文件路径
PROCESS_TEST_DATA_PATH = os.path.join(BASE_DIR, "data/processData/test") # 预处理后的测试文件路径
STOPWORDS_PATH = os.path.join(BASE_DIR, "data/stopWords/stopWords.txt")

# 生成文件夹函数,生成PROCESS_TRAIN_DATA_PATH，PROCESS_TEST_DATA_PATH
def genFolder(folderPath):
    if os.path.exists(folderPath):
        pass
    else:
        os.mkdir(folderPath)
genFolder(PROCESS_TRAIN_DATA_PATH)
genFolder(PROCESS_TEST_DATA_PATH)
# 生成停止词列表
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    STOPWORD_LIST = f.read().splitlines()
