from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, chi2, mutual_info_classif, f_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
# from sklearn.feature_extraction import
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from scipy import sparse
import math
import time

class Feature_Selector:
    def __init__(self, trainBunch, testBunch):
        self.trainBunch = trainBunch   # labels是一个列表类型，每一个元素均是列表，该列表包含该文本的标签，例如[[2],[1]]
        self.testBunch = testBunch    # contents也是一个列表类型，每一个元素均是一个列表，该列表包含的是该文本分伺后的词语，例如[["我 爱 北京 天安门"],["..."]]
    # 生成所需的参数
    def gen_parameters(self): # 对于自定义方法需要生成一些参数。
        # 生成的参数有vectors, Metrix, word_dict, class_dict, class_document_number, A, B, C, D
        print("正在生成词频矩阵》》》》")
        st = time.time()
        word2vector = CountVectorizer(max_df=0.5)
        self.trainBunch.vectors = word2vector.fit_transform(self.trainBunch.contents, self.trainBunch.labels)  # 获得词频矩阵，vector储存词频矩阵
        self.testBunch.vectors = word2vector.transform(self.testBunch.contents)  # 获得词频矩阵
        self.trainBunch.word_dict_ = word2vector.vocabulary_   # 单词字典, 最终生成{"我"：0,...}的形式
        self.testBunch.word_dict_ = word2vector.vocabulary_    # 单词字典, 最终生成{"我"：0,...}的形式
        et = time.time()
        print("已生成词频矩阵！！！！用时为{:.3f}s。".format(et - st))
        print("正在生成特征选择所需参数》》》》")
        print("正在提取单词字典》》》》")
        # 下面的参数用于选择特征所以只需要计算trainBunch的参数即可。注意self.trainBunch.word_dict_和self.trainBunch.word_dict的区别，self.trainBunch.word_dict_是CountVectorizer()生成的，用于后续处理稀疏矩阵。
        # self.trainBunch.word_dict是自己统计出来的，用于生成后面的文档矩阵等参数，两者稍有不同但不影响。
        self.trainBunch.word_dict = {}  # word_dict单词字典, 最终生成{"我"：0,...}的形式, 自己写的提取单词的程序，要比CountVectorizer()生成的要多一些
        self.trainBunch.index_word = {}  # index_word 用于反向找到word
        st = time.time()
        index = 0
        for line in self.trainBunch.contents: # line是一个文档的单词列表， 遍历整个训练集
            for word in line.split(" "):  # word为一个单词
                if word in self.trainBunch.word_dict.keys():
                    pass
                else:
                    self.trainBunch.word_dict[word] = index
                    index += 1
        for word, index in self.trainBunch.word_dict.items():
            self.trainBunch.index_word[index] = word
        et = time.time()
        print("已生成单词字典！！！！用时为{:.3f}s。".format(et-st))
        print("正在提取类别字典》》》》")
        st = time.time()
        self.trainBunch.class_dict = {}  # 类别字典，最终生成{类别1:0,...}的形式
        index = 0
        for label in self.trainBunch.labels: # lable是一个文档类别， 遍历整个训练集
            if label in self.trainBunch.class_dict.keys():
                continue
            else:
                self.trainBunch.class_dict[label] = index
                index += 1
        et = time.time()
        print("已生成类别字典！！！！用时为{:.3f}s。".format(et-st))
        print("正在生成文档频率矩阵等参数》》》》")
        st = time.time()
        train_file_num = len(self.trainBunch.labels)  # 训练集全部的文档数目
        row = len(self.trainBunch.word_dict.keys())  # 行为单词数量
        column = len(self.trainBunch.class_dict.keys())  # 列为文档类别
        self.trainBunch.A = np.zeros(shape=(row, column))  # A为二维数组，aij表示包含特征词i，类别属于j的文档数量
        self.trainBunch.class_document_number = np.zeros(shape=(1, column), dtype=np.int16)  # class_document_number为一维数组， 第j个元素表示j类别文档的数量
        for label, line in zip(self.trainBunch.labels, self.trainBunch.contents):  # label是文档标签，line为该文档的内容
            column_index = self.trainBunch.class_dict[label]  # 列为类别索引
            self.trainBunch.class_document_number[0][column_index] += 1  # 文档计数
            line_index = [self.trainBunch.word_dict[word] for word in line.split(" ")]  # 将单词转化为索引数字
            line_set = set(line_index)  # 去除重复的单词索引，一篇文档中，单词无论出现多少次，aij的计数都只加1。
            for word_index in line_set:
                self.trainBunch.A[word_index][column_index] += 1
        self.trainBunch.B = np.array([(sum(x) - x).tolist() for x in self.trainBunch.A])  # B为二维数组，bij表示包含特征词i，类别不属于j的文档数量
        self.trainBunch.C = np.tile(self.trainBunch.class_document_number, (self.trainBunch.A.shape[0], 1)) - self.trainBunch.A  # C为二维数组，cij表示不包含特征词i，类别属于j的文档数量
        self.trainBunch.N = np.full((row, column), train_file_num)  # N为二维数组，每一个元素表示所有文档的数量
        self.trainBunch.D = self.trainBunch.N - self.trainBunch.A - self.trainBunch.B - self.trainBunch.C  # D为二维数组，dij表示不包含特征词i，类别也不属于j的文档数量
        self.trainBunch.class_set_size = len(self.trainBunch.class_dict.keys())  # class_set_size 表示文档一共有多少类
        self.trainBunch.term_df_array = np.sum(self.trainBunch.A, axis=1)  # 词i一共出现在多少文档里
        self.trainBunch.C_Total = np.tile(self.trainBunch.class_document_number, (self.trainBunch.A.shape[0], 1))  # 每类一共多少文档
        self.trainBunch.C_Total_Not = self.trainBunch.N - self.trainBunch.C_Total  # j类之外一共有多少文档。
        self.trainBunch.term_set_size = len(self.trainBunch.word_dict.keys())  # 一共多少不重复的特征
        et = time.time()
        print("所有参数生成完毕！！！！用时为{:.3f}s。".format(et-st))
        return self.trainBunch, self.testBunch
    # 基于互信息的特征选择方法(自定义)
    def mi(self, num=None, per=None): # num和per有且只有一个不为空, 倾向于低频词，效果不是很好
        st = time.time()
        mi_metrix = np.log(((self.trainBunch.A + 1.0) * self.trainBunch.N) / ((self.trainBunch.A + self.trainBunch.C) * (self.trainBunch.A + self.trainBunch.B + self.trainBunch.class_set_size))) # Xij表示第i个词对于j类文档的互信息值
        mi_max_list = [max(x) for x in mi_metrix]  # 选出每个词对所有不同类别文档互信息的最大值
        mi_score_array = np.array(mi_max_list)  # 转化为数组
        sorted_score_index = mi_score_array.argsort()[:: -1]  # 排列出数值从大到小的索引
        self.trainBunch,self.testBunch = self.compute_mi_ig_wllr_select_features(sorted_score_index,st,num,per)
        return self.trainBunch, self.testBunch
    # 基于信息增益的特征选择(自定义)
    def ig(self, num=None, per=None): # num和per有且只有一个不为空， 实际上是互信息的加权
        st = time.time()
        p_t = self.trainBunch.term_df_array / self.trainBunch.N[0][0]
        p_not_t = 1 - p_t
        p_c_t_mat = (self.trainBunch.A + 1) / (self.trainBunch.A + self.trainBunch.B + self.trainBunch.class_set_size)
        p_c_not_t_mat = (self.trainBunch.C + 1) / (self.trainBunch.C + self.trainBunch.D + self.trainBunch.class_set_size)
        p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
        p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)
        ig_score_array = p_t * p_c_t + p_not_t * p_c_not_t
        sorted_score_index = ig_score_array.argsort()[:: -1] # 排列出数值从大到小的索引
        self.trainBunch, self.testBunch = self.compute_mi_ig_wllr_select_features(sorted_score_index, st, num, per)
        return self.trainBunch, self.testBunch
    # 基于wllr的特征选择(自定义)
    def wllr(self, num=None, per=None):  # num和per有且只有一个不为空
        p_t_c = (self.trainBunch.A + 1E-6) / (self.trainBunch.C_Total + 1E-6 * self.trainBunch.term_set_size)
        st = time.time()
        p_t_not_c = (self.trainBunch.B + 1E-6) / (self.trainBunch.C_Total_Not + 1E-6 * self.trainBunch.term_set_size)
        term_score_mat = p_t_c * np.log(p_t_c / p_t_not_c)
        wllr_score_max_list = [max(x) for x in term_score_mat]
        wllr_score_array = np.array(wllr_score_max_list)
        sorted_score_index = wllr_score_array.argsort()[:: -1]  # 排列出数值从大到小的索引
        self.trainBunch, self.testBunch = self.compute_mi_ig_wllr_select_features(sorted_score_index, st, num, per)
        return self.trainBunch, self.testBunch

    def compute_mi_ig_wllr_select_features(self,sorted_score_index, st, num=None, per=None):  # 用于计算自定义方法的通用框架
        if num:  # num是需要选择的特征数，应为正整数。
            print("正在进行根据自定义互信息算法的特征选择，特征选择数目为{:d}个".format(num))
            select_index_array = sorted_score_index[:num]  # 根据word_dict获得的索引
            select_word_list = [self.trainBunch.index_word[index] for index in select_index_array]  # 反向获得word
            select_index_list = []   # 获得word对应的word_dict_的索引
            for word in select_word_list:
                try:
                    select_index_list.append(self.trainBunch.word_dict_[word])
                except:
                    select_index_list.append(0)
            self.trainBunch.select_features = sparse.csr_matrix(self.trainBunch.vectors[:, select_index_list[0]])  # 获得选择后的特征稀疏矩阵
            for index in select_index_list[1:]:
                self.trainBunch.select_features = sparse.hstack((self.trainBunch.select_features, self.trainBunch.vectors[:, index]))
            self.testBunch.select_features = sparse.csr_matrix(self.testBunch.vectors[:, select_index_list[0]])  # 获得选择后的特征稀疏矩阵
            for index in select_index_list[1:]:
                self.testBunch.select_features = sparse.hstack((self.testBunch.select_features, self.testBunch.vectors[:, index]))
            et = time.time()
        elif per:  # per是需要选择的特征数占全部特征的比例，范围0~100
            print("正在进行根据自定义互信息算法的特征选择，特征选择比例为前{:.3f}%".format(per/100.0))
            select_features_number = int(len(self.trainBunch.word_dict.keys()) * per / 100)  # 选取的特征数目
            select_index_array = sorted_score_index[:select_features_number]  # 根据word_dict获得的索引
            select_word_list = [self.trainBunch.index_word[index] for index in select_index_array]  # 反向获得word
            select_index_list = []  # 获得word对应的word_dict_的索引
            for word in select_word_list:
                try:
                    select_index_list.append(self.trainBunch.word_dict_[word])
                except:
                    select_index_list.append(0)
            self.trainBunch.select_features = sparse.csr_matrix(
                self.trainBunch.vectors[select_index_list[0]])  # 获得选择后的特征稀疏矩阵
            for index in select_index_list[1:]:
                self.trainBunch.select_features = sparse.hstack(
                    (self.trainBunch.select_features, self.trainBunch.vectors[select_index_list[index]]))
            self.testBunch.select_features = sparse.csr_matrix(
                self.testBunch.vectors[select_index_list[0]])  # 获得选择后的特征稀疏矩阵
            for index in select_index_list[1:]:
                self.testBunch.select_features = sparse.hstack(
                    (self.testBunch.select_features, self.testBunch.vectors[select_index_list[index]]))
            et = time.time()
        else:
            print("未输入所需参数！")
        print("特征选择完成！！！！用时：{:.3f}s。".format(et - st))
        return self.trainBunch, self.testBunch
    # 基于互信息的特征选择方法（sklearn接口版本）
    def mutual_info(self, num=None, per=None):  # num和per有且只有一个不为空
        self.trainBunch, self.testBunch = self.compute_mutualinfo_chi2_fclassif_select_features(mutual_info_classif, num, per)
        return self.trainBunch, self.testBunch
    # 基于卡方的特征选择方法， 判断词与文档是否相关
    def chi2(self, num=None, per=None): # num和per有且只有一个不为空
        self.trainBunch, self.testBunch = self.compute_mutualinfo_chi2_fclassif_select_features(chi2, num, per)
        return self.trainBunch, self.testBunch
    # 基于方差分析的特征选测方法
    def f_classif(self, num=None, per=None):  # num和per有且只有一个不为空
        self.trainBunch, self.testBunch = self.compute_mutualinfo_chi2_fclassif_select_features(f_classif, num, per)
        return self.trainBunch, self.testBunch

    def compute_mutualinfo_chi2_fclassif_select_features(self, model, num=None, per=None):  # 计算sklearn接口方法的通用框架。
        st = time.time()
        if num:  # num是需要选择的特征数，应为正整数。
            print("正在进行根据互信息的特征选择，特征选择数目为{:d}个".format(num))
            selector = SelectKBest(model, num)
            self.trainBunch.select_features = selector.fit_transform(self.trainBunch.vectors, self.trainBunch.labels)
            self.testBunch.select_features = selector.transform(self.testBunch.vectors)
            et = time.time()
        elif per:  # per是需要选择的特征数占全部特征的比例，范围0~100
            print("正在进行根据互信息的特征选择，特征选择比例为前{:.3f}%".format(per / 100.0))
            selector = SelectPercentile(model, per)
            self.trainBunch.select_features = selector.fit_transform(self.trainBunch.vectors, self.trainBunch.labels)
            self.testBunch.select_features = selector.transform(self.testBunch.vectors)
            et = time.time()
        else:
            print("未输入所需参数！")
            et = time.time()
        print("特征选择完成！！！！用时：{:.3f}s。".format(et - st))
        return self.trainBunch, self.testBunch  # 选择后的词频矩阵

    # 基于tfidf值的特征选择, 自定义的特征选择方法
    def compute_select_features_tfidf(self): # 计算选取好的特征的tfidf值
        st = time.time()
        print("开始计算所选特征的tfidf值》》》》")
        try:
            selector = TfidfTransformer(sublinear_tf=True)
            self.trainBunch.select_tfidf = selector.fit_transform(self.trainBunch.select_features, self.trainBunch.labels)
            self.testBunch.select_tfidf = selector.transform(self.testBunch.select_features)
            et = time.time()
            print("计算所选特征的tfidf值完成！！！！用时：{:.3f}s。".format(et - st))
            return self.trainBunch, self.testBunch
        except:
            print("尚未进行特征选择！！！！")
    # 基于tfidf值的特征选择, 自定义的特征选择方法
    def compute_all_features_tfidf(self): # 计算全部特征的tfidf值
        st = time.time()
        print("开始计算全部特征的tfidf值》》》》")
        selector = TfidfTransformer(sublinear_tf=True)
        self.trainBunch.tfidf = selector.fit_transform(self.trainBunch.vectors, self.trainBunch.labels)
        self.testBunch.tfidf = selector.transform(self.testBunch.vectors)
        et = time.time()
        print("计算全部特征的tfidf值完成！！！！用时：{:.3f}s。".format(et - st))
        return self.trainBunch, self.testBunch
    # 基于模型的特征选择
    # 1.基于LinearSVC模型的特征选择
    def model_LinearSVC(self, C=0.01):  # C控制特征选择的数量，C越小特征选择的个数越少
        st = time.time()
        print("开始进行LinearSVC模型预训练》》》》")
        model = LinearSVC(C=C, penalty="l1", dual=False).fit(self.trainBunch.vectors, self.trainBunch.labels)
        print("LinearSVC模型预训练完毕，开始选择特征》》》》")
        selector = SelectFromModel(model, prefit=True)
        self.trainBunch.select_features = selector.transform(self.trainBunch.vectors)
        self.testBunch.select_features = selector.transform(self.testBunch.vectors)
        et = time.time()
        print("特征选择完成！！！！用时：{:.3f}s。总共选择特征为{:d}个！！！！".format((et - st), self.trainBunch.select_features.shape[1]))
        return self.trainBunch, self.testBunch  # trainBunch.select_features,testBunch.select_features为选择后的词频矩阵
    # 2.基于决策树的特征选择
    def model_decisiontree(self, n_estimators=10):  # n_estimators控制特征选择的数量，n_estimators越小特征选择的个数越少
        st = time.time()
        print("开始进行决策树模型预训练》》》》")
        model = ExtraTreesClassifier(n_estimators=n_estimators).fit(self.trainBunch.vectors, self.trainBunch.labels)
        print("决策树模型预训练完毕，开始选择特征》》》》")
        selector = SelectFromModel(model, prefit=True)
        self.trainBunch.select_features = selector.transform(self.trainBunch.vectors)
        self.testBunch.select_features = selector.transform(self.testBunch.vectors)
        et = time.time()
        print("特征选择完成！！！！用时：{:.3f}s。总共选择特征为{:d}个！！！！".format((et - st), self.trainBunch.select_features.shape[1]))
        return self.trainBunch, self.testBunch  # trainBunch.select_features,testBunch.select_features为选择后的词频矩阵
