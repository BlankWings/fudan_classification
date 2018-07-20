# 主函数完成对预处理后文件的特征选择，特征权重计算以及训练预测
from helper import *
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif, RFE, RFECV, f_classif
from sklearn import metrics
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import re
from feature_selector import Feature_Selector

# 获得训练样本和测试样本内容内容以及标签，储存在bunch之中, 也就是labels和contents
def genBunch(processDataPath):  # processDataPath类似与"../data/processData/train"
    bunch = Bunch(labels=[], contents=[])
    subfileList = os.listdir(processDataPath) # 类似于[...,C11-space.txt,...]
    for subfile in subfileList: # subfile 为C1-space.txt
        label = int(re.sub("[^\d]","",subfile)) # 提取数字标签
        subfileName = os.path.join(processDataPath, subfile)
        with open(subfileName, "r", encoding="utf-8") as f:
            for content in f.readlines():
                lenth = len(content.split(" "))
                if lenth > 20:  # 过滤掉文本长度小于20的数据
                    bunch.labels.append(label)
                    bunch.contents.append(content.strip())
    return bunch


# 对特征选择后的databunch进行模型训练，返回预测结果, 配合多进程使用
def clf2result(select_feature_number):
    #trainDatabunch是已生成所需参数的trainDatabunch, testDatabunch是已生成所需参数的testDatabunch, selection_method是特征选择方法, clf_method分类器方法。
    trainBunch, testBunch = feature_select.chi2(num=select_feature_number)
    trainBunch, testBunch = feature_select.compute_select_features_tfidf()
    clf_model.fit(trainBunch.select_tfidf, trainBunch.labels)
    predict_y = clf_model.predict(testBunch.select_tfidf)
    F1 = metrics.f1_score(testBunch.labels, predict_y, average='weighted')
    F1 = eval("{:.3f}".format(F1))  # 截取小数点后3位
    return F1

# 对分类结果产生的混淆矩阵可视化
def plot_cfm(cm, title="Confusion matrix", cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_LIST))
    plt.xticks(tick_marks, CLASS_LIST, rotation=45)
    plt.yticks(tick_marks, CLASS_LIST)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
# 绘制不同特征选择数的结果，title为分类器名称
def plot_result(result_dict, title):
    x_list = range(len(result_dict.keys()))
    feature_number = list(result_dict.keys())
    f1_value = result_dict.values()
    plt.figure(figsize=(16, 24))
    plt.plot(x_list, f1_value, color='r',markerfacecolor='blue',marker='o')
    plt.xlabel("select_features")
    plt.xticks([])
    plt.ylabel("f1_value")
    plt.title(title)
    for x,y  in zip(x_list, f1_value):
        plt.text(x,y,(feature_number[x],y),ha="center",va="bottom")
    plot_file = os.path.join(PLOT_FILE, title + ".jpg")
    plt.savefig(plot_file) # 保存图片
    plt.show()

if __name__ == '__main__':
    '''
    # 获取bunch中的labels和contents
    trainBunch = genBunch(PROCESS_TRAIN_DATA_PATH)  # Bunch中的labels储存标签，contents存储文本内容，vector储存词频矩阵，selectVector储存特征选择后的词频矩阵
    testBunch = genBunch(PROCESS_TEST_DATA_PATH)    # tfidf储存由vector生成的tfidf矩阵，selectTfidf储存由selectVector生成或者tfidf特征选择得到的最终的tfidf矩阵
    # 生成所需参数，并储存trainBunch,testBunch
    feature_select = Feature_Selector(trainBunch, testBunch)
    trainBunch, testBunch = feature_select.gen_parameters()
    joblib.dump(trainBunch, TRAIN_BUNCH_FILE)
    joblib.dump(testBunch, TEST_BUNCH_FILE)
    '''
    # 加载trainBunch和testBunch
    print("正在进行准备工作》》》》")
    trainBunch = joblib.load(TRAIN_BUNCH_FILE)
    testBunch = joblib.load(TEST_BUNCH_FILE)
    feature_select = Feature_Selector(trainBunch, testBunch) # 实例化类

    # 特征哈希
    # from sklearn.svm import LinearSVC
    # print(trainBunch.vectors.shape)
    # from sklearn.feature_extraction.text import HashingVectorizer
    # haxi = HashingVectorizer(n_features=2**16, non_negative=True)
    # trainBunch.haxi = haxi.transform(trainBunch.contents)
    # testBunch.haxi = haxi.transform(testBunch.contents)
    # print(trainBunch.haxi.shape)
    # result_dict = {}
    # for n_f in [100, 200, 500, 1000, 2000,5000, 10000, 20000,50000, 65536]:
    #     select = SelectKBest(chi2, n_f)
    #     trainBunch.select_haxi = select.fit_transform(trainBunch.haxi, trainBunch.labels)
    #     testBunch.select_haxi = select.transform(testBunch.haxi)
    #     tfidf  = TfidfTransformer(sublinear_tf=True)
    #     trainBunch.select_tfidf = tfidf.fit_transform(trainBunch.select_haxi, trainBunch.labels)
    #     testBunch.select_tfidf = tfidf.transform(testBunch.select_haxi)
    #     model = LinearSVC()
    #     model.fit(trainBunch.select_tfidf, trainBunch.labels)
    #     predict_y = model.predict(testBunch.select_tfidf)
    #     F1 = metrics.f1_score(testBunch.labels, predict_y, average='weighted')
    #     F1 = eval("{:.3f}".format(F1))  # 截取小数点后3位
    #     result_dict[n_f] = F1
    # plot_result(result_dict, "haxi")


    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf_model = LogisticRegression(solver="saga", multi_class="multinomial")  # 定义分类器

    all_feature_num = trainBunch.vectors.shape[1]
    select_feature_number_list = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, all_feature_num]  # 特征选择的数量
    result_dict = {}
    print("准备工作完成！！！！")
    # 测试不同的特征选择方法。
    print("正在进行特征选择》》》》")
    pool = multiprocessing.Pool(processes=12)
    for select_feature_number in select_feature_number_list:
        result_dict[select_feature_number] = pool.map(clf2result, (select_feature_number,))[0]   # map方法好像不太能利用多核优势, 这个不是真正的多进程。
    pool.close()
    pool.join()
    plot_result(result_dict,"LogisticRegression")




