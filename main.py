# 主函数完成对预处理后文件的特征选择，特征权重计算以及训练预测
from helper import *
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.datasets.base import Bunch
import pandas as pd

# 获得训练样本和测试样本内容内容以及标签，储存在bunch之中
def genBunch(processDataPath):  # processDataPath类似与"../data/processData/train"
    bunch = Bunch(labels=[], contents=[], tfidf=[])
    subfileList = os.listdir(processDataPath)
    for subfile in subfileList:
        label = [eval(subfile.split("C")[1].split("-")[0])]# 提取数字标签
        subfileName = os.path.join(processDataPath, subfile)
        with open(subfileName, "r", encoding="utf-8") as f:
            for content in f.readlines():
                lenth = len(content.split(" "))
                if lenth > 20:
                    bunch.labels.append(label)
                    bunch.contents.append(content.strip())
    return bunch

def counterWord(dataBunch):
    contents = []
    for content in dataBunch.contents:
        lenth = len(content.split(" "))
        for word in content.split(" "):
            contents.append(word)

    print(len(contents))
    print(len(set(contents)))



if __name__ == '__main__':
    trainBunch = genBunch(PROCESS_TRAIN_DATA_PATH)
    testBunch = genBunch(PROCESS_TEST_DATA_PATH)

    # vector = CountVectorizer(max_df=0.6,ngram_range=(1,2)) # 添加了2gram特征，结果表明准确率下降了
    vector = CountVectorizer(max_df=0.6)
    tfidfTransfor = TfidfTransformer(sublinear_tf=True)
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.6)

    print("正在生成tfidf空间。。。")
    trainVerctorMetrix = vector.fit_transform(trainBunch.contents)
    testVectorMetrix = vector.transform(testBunch.contents)
    #进行特征选择
    from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
    selection_method = SelectPercentile(chi2, percentile=10)
    newTrainVerctorMetrix = selection_method.fit_transform(trainVerctorMetrix, trainBunch.labels)
    trainBunch.tfidf = tfidfTransfor.fit_transform(newTrainVerctorMetrix)
    new_test_verctor_metrix = selection_method.transform(testVectorMetrix)
    testBunch.tfidf = tfidfTransfor.transform(new_test_verctor_metrix)


    #
    # testBunch.tfidf = tfidfTransfor.transform(testVectorMetrix)
    # # trainBunch.tfidf = tfidf.fit_transform(trainBunch.contents)
    # # testBunch.tfidf = tfidf.transform(testBunch.contents)
    print("生成tfidf空间完毕！！！")



    print("正在进行模型训练。。。")
    model = LinearSVC()
    model.fit(trainBunch.tfidf, trainBunch.labels)
    predict = model.predict(testBunch.tfidf)
    print("模型训练完成！！！")
    f1 = metrics.f1_score(y_true=testBunch.labels, y_pred=predict, average='weighted') # 多分类要加入average='weighted'
    print("最终结果的F值为：{:.3f}".format(f1))


