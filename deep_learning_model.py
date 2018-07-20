# 使用keras进行文本分类
# coding=utf-8
# 使用的网络结构包含MLP， RNN， LSTM, GRU, CNN等。
from helper import *
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import time, re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Input, concatenate
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import Bidirectional  # 构建双向循环网络
from keras.models import Model
import numpy as np


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

def MLP_NETWORK():  # 多层感知机神经网络结构
    model = Sequential()   # 堆叠式模型
    # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
    model.add(Embedding(input_dim=token_words, input_length=max_lenth, output_dim=32))
    model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
    model.add(Flatten())     # 加入Flattern层，变为３２００个神经元
    model.add(Dense(units=256, activation="relu"))  # 隐藏层
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation="softmax"))  # 输出层
    print("神经网络的结构如下：")
    print(model.summary())
    return model
def RNN_NETWORK():
    model = Sequential()   # 堆叠式模型
    # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
    model.add(Embedding(input_dim=token_words, input_length=max_lenth, output_dim=32))
    model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
    model.add(SimpleRNN(units=16))     # 加入Flattern层，变为３２００个神经元
    model.add(Dense(units=256, activation="relu"))  # 隐藏层
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation="softmax"))  # 输出层
    print("神经网络的结构如下：")
    print(model.summary())
    return model
def BIRNN_NETWORK():
    model = Sequential()   # 堆叠式模型
    # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
    model.add(Embedding(input_dim=token_words, input_length=max_lenth, output_dim=32))
    model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
    model.add(Bidirectional(SimpleRNN(units=16, return_sequences=True), merge_mode="concat"))     # 加入Flattern层，变为３２００个神经元
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))  # 隐藏层
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation="softmax"))  # 输出层
    print("神经网络的结构如下：")
    print(model.summary())
    return model
def LSTM_NETWORK():
    model = Sequential()   # 堆叠式模型
    # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
    model.add(Embedding(input_dim=token_words, input_length=max_lenth, output_dim=32))
    model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
    model.add(LSTM(units=32))     # 加入Flattern层，变为３２００个神经元
    model.add(Dense(units=256, activation="relu"))  # 隐藏层
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation="softmax"))  # 输出层
    print("神经网络的结构如下：")
    print(model.summary())
    return model
def GRU_NETWORK():
    model = Sequential()   # 堆叠式模型
    # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
    model.add(Embedding(input_dim=token_words, input_length=max_lenth, output_dim=32))
    model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
    model.add(GRU(units=32))     # 加入Flattern层，变为３２００个神经元
    model.add(Dense(units=256, activation="relu"))  # 隐藏层
    model.add(Dropout(0.5))
    model.add(Dense(units=20, activation="softmax"))  # 输出层
    print("神经网络的结构如下：")
    print(model.summary())
    return model
def TEXT_CNN_NETWORK(): # 还有些方法不是很懂
    sentence_seq = Input(shape=[max_lenth], name="X_seq")  # 输入
    embedding_layer = Embedding(input_dim=token_words, output_dim=32)(sentence_seq)  # 词嵌入层
    # 卷积层如下：
    convs = []
    filter_sizes = [2, 3, 4, 5]  # 4种卷积核
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation="relu")(embedding_layer)    # 100个(max_lenth-fsz+1, 1)维的向量
        l_pool = MaxPooling1D(max_lenth-fsz+1)(l_conv)    # 100个1维向量
        l_pool = Flatten()(l_pool)   # 拉平
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.35)(merge)
    output = Dense(512, activation="relu")(out)
    output = Dropout(0.35)(output)
    output = Dense(256, activation="relu")(output)
    output = Dense(20, activation="softmax")(output)
    model = Model([sentence_seq], output)
    print("神经网络的结构如下：")
    print(model.summary())
    return model

if __name__ == '__main__':
    '''
    # 获取bunch中的labels和contents
    print("正在获取数据》》》》")
    st = time.time()
    trainBunch = genBunch(PROCESS_TRAIN_DATA_PATH)  # Bunch中的labels储存标签，contents存储文本内容，vector储存词频矩阵，selectVector储存特征选择后的词频矩阵
    testBunch = genBunch(PROCESS_TEST_DATA_PATH)    # tfidf储存由vector生成的tfidf矩阵，selectTfidf储存由selectVector生成或者tfidf特征选择得到的最终的tfidf矩阵
    joblib.dump(trainBunch, TRAIN_BUNCH_FILE_DL)
    joblib.dump(testBunch, TEST_BUNCH_FILE_DL)
    et = time.time()
    print("获取并保存数据成功！！！！用时{:.3f}s。".format(et-st))
    '''

    print("正在加载数据》》》》")
    trainBunch = joblib.load(TRAIN_BUNCH_FILE)
    testBunch = joblib.load(TEST_BUNCH_FILE)
    print("加载数据成功！！！！")
    # 将labels处理为20维的one_hot向量
    transformer = MultiLabelBinarizer()
    trainBunch.labels_list = [[trainBunch.labels[i]] for i in range(len(trainBunch.labels))]  # 将label变为[[11], [11], [11], [11], [11], [11], [11], ...]才能进行one_hot向量化。
    testBunch.labels_list = [[testBunch.labels[i]] for i in range(len(testBunch.labels))]  # 将label变为[[11], [11], [11], [11], [11], [11], [11], ...]才能进行one_hot向量化。
    trainBunch.new_labels = transformer.fit_transform(trainBunch.labels_list)
    testBunch.new_labels = transformer.transform(testBunch.labels_list)
    # 相关参数如下：
    token_words = 3800   # 单词字典的单词数。
    max_lenth = 380      # 选取句子的长度。
    print("正在进行数据预处理》》》》")
    st = time.time()
    # 建立Token词典
    token = Tokenizer(num_words=token_words)  # 设置词典规模
    token.fit_on_texts(trainBunch.contents)   # 建立字典模型
    # 将文字列表转化为数字列表
    trainBunch.contents_seq = token.texts_to_sequences(trainBunch.contents)
    testBunch.contents_seq = token.texts_to_sequences(testBunch.contents)
    # 对数字列表进行padding，截长补短。处理后的数据输入神经网络进行训练。
    trainBunch.contents_seq_pad = sequence.pad_sequences(trainBunch.contents_seq, maxlen=max_lenth)
    testBunch.contents_seq_pad = sequence.pad_sequences(testBunch.contents_seq, maxlen=max_lenth)
    et = time.time()
    print("数据预处理完成！！！！用时：{:.3f}s".format(et-st))
    # 构建神经网络，
    # model = MLP_NETWORK()
    # model = RNN_NETWORK()
    # model = LSTM_NETWORK()
    # model = GRU_NETWORK()
    # model = BIRNN_NETWORK()
    
    model = TEXT_CNN_NETWORK()
    # 定义训练方法
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # 打乱数据集
    shuffle_index = [i for i in range(len(trainBunch.new_labels))]
    np.random.shuffle(shuffle_index)
    trainBunch.new_labels = trainBunch.new_labels[shuffle_index]
    trainBunch.contents_seq_pad = trainBunch.contents_seq_pad[shuffle_index]

    # 开始训练
    train_history = model.fit(trainBunch.contents_seq_pad, trainBunch.new_labels, batch_size=100, epochs=10, verbose=1, validation_split=0.2)
    # 评估模型准确率
    score = model.evaluate(testBunch.contents_seq_pad, testBunch.new_labels, verbose=2)
    print("模型准确率为：{:.3f}".format(score[1]))

    # 保存模型和Token字典
    model.save(DL_MODEL_FILE)
    joblib.dump(token, TOKEN_MODEL_FILE)
    print("保存模型完成！！！")


    # 下面是验证过程
    print("正在加载模型和token字典")
    my_model = load_model(DL_MODEL_FILE)
    my_token = joblib.load(TOKEN_MODEL_FILE)

    sentence = ["日期 版号 标题 教育 子女 家庭 和 社会 的 共同 责任 家庭 伦理 与 人生 幸福 讨论 生活 话题 作者 闻佳平 吴 鲲鹏 程晋舜 陈家彬 樊天林 郑 保健 凌传昌 正文 主持人 本版 编辑 闻佳平 主持人 家庭 是 人生 的 摇篮 这句 话 不仅仅 是 指 家庭 是 一个 人 出生 成长 的 地方 同时 也 意味着 家庭 是 一个 人 思想道德 和 价值观念 形成 的 起点 家庭 担负 着 养育 和 教育 子女 的 双重 职能 随着 经济 的 发展 物质财富 的 不断 增长 养育 孩子 对 绝大多数 家庭 来说 已经 不是 什么 大 问题 了 但是 我们 应该 怎样 教育 子女 把 具备 什么 素质 的 子女 输向 社会 却是 每个 家庭 都 在 思考 和 应该 思考 的 大 问题 因为 子女 是 家庭 和 社会 的 希望 相互尊重 子女 的 期盼 主持人 从 血缘 角度看 子女 是 父母 爱情 的 结晶 但 从 法律 角度 讲 子女 又 是 具有 独立 人格 的 个体 因此 父母 与 子女 之间 既有 血缘 亲情 又 有 平等 人格 要 处理 好 这 二者之间 的 关系 一个 基本 的 做法 是 相互尊重 吴 鲲鹏 湖北 鄂州市 农机 公司 我 刚刚 走上 工作岗位 已 不 在 孩子 的 行列 但 我 却 有 太 多 的话 想替 天下 的 孩子 对 天下 的 父母 们 说 这太多 的话 浓缩 成 四个 字 就是 尊重 孩子 在 今天 父母 与 未成年 子女 的 关系 问题 正 逐渐 成为 一个 难题 越来越 多 的 父母 抱怨 自己 的 孩子 不听话 不争气 而 孩子 们 则 指责 父母 们 太 专制 太守 旧 不过 中国 有句 古话 养不教 父之过 一块 璞玉 若 不能 成器 玉工 恐怕 难辞其咎 在 教育 孩子 问题 上 一方面 很多 父母 们 对 孩子 的 管教 方式 仍 未 摆脱 棍棒 之下 出 孝子 的 传统 方法 往往 用 威压 的 手段 在 家庭 中 树立 自己 的 权威 凡事 命令 强迫 的 多 讲道理 的 少 更 谈不上 征求 孩子 的 意见 这种 管教 方式 虽然 暂时 使 子女 做 了 父母 心目 中 的 乖孩子 但 随着 年龄 的 增长 孩子 们 自我意识 增强 不再 像 以往 那般 驯服 而 父母 们 发现自己 的 权威 开始 动摇 时 往往 采取 更 粗暴 的 手段 来 逼 孩子 就范 结果 要么 是 使 孩子 变得 怯懦 失去 自信心 和 创造力 要么 是 激起 孩子 的 逆反心理 另一方面 近几年 溺爱 成为 许多 父母 的 管教 方式 随着 独生子女 家庭 增多 年轻 父母 们 一切 以 孩子 为 中心 对 孩子 的 要求 千方百计 予以 满足 一大批 小 公主 小 皇帝 涌现 出来 许多 孩子 自我意识 膨胀 变得 自私 而 蛮横 却 又 依赖性 强 威压 和 溺爱 的 管教 方式 都 是 对 孩子 的 不 尊重 只不过 一个 是 显性 的 一个 是 隐性 的 前者 完全 忽视 孩子 的 人格 权利 后者 则 过于 轻率地 将 这 一 权利 扭曲 成为 一种 间接 的 不 尊重 程晋舜 安徽 淮南市 潘集区 几乎 和 所有 的 老三届 一样 我 为 儿子 设计 了 一条 读 高中 上 大学 拿 学位 的 路 企望 通过 孩子 的 努力 弥补 当年 自己 的 缺憾 住房 不大 也 专门 为 他 辟 一间 书房 薪水 不高 却 从不 在 买 学习 资料 上 吝啬 我们 夫妻俩 早晨 一碗 泡饭 孩子 却是 煎蛋 豆奶 然而 初中 一 毕业 儿子 却 告诉 我爸 我 想 当兵 我 不 反对 青年人 参军 但 潜意识 里 却 认为 上 大学 才 有利于 实现 人 的 价值 没 受过 高等教育 毕竟 是 人生 一大 缺憾 一向 幼稚 仅 有 岁 的 儿子 冷静 得 叫 我 吃惊 他 认为 自己 学习成绩 一般 在 考大学 的 竞争 中 没有 优势 即使 考取 大学 老爸 老妈 又 要 四处 借贷 筹措 学费 要 为 儿子 的 每 一步 奋斗 操碎心 儿子 随着 军列 走 了 他 没 让 我们 失望 新兵 三个 月 训练 他成 了 特等 射手 学习 标兵 训练 标兵 连获 三次 嘉奖 前不久 他 参加 军事 会操 时 头部 负伤 流着 血 坚持 表演 到 最后 我 到 部队 看 他 时 他 说 爸 你 儿子 是 军人 流点 血算 啥 我 说 儿子 你 真的 长大 了 他 想 考 军校 征求 我 的 意见 我 想 了 想 郑重 地 回答 他 我 尊重 你 的 选择 主持人 谈到 目前 子女 与 父母 之间 的 关系 人们 爱 用 代沟 这个 词来 概括 从 社会学 角度 讲 代沟 是 指 处于 不同 代际 的 群体 或 个体 间 沟通 上 的 障碍 一般而言 代沟 往往 发生 在 社会变迁 激烈 的 时期 中国 和 整个 世界 都 正 处于 大 变革 的 时代 科学技术 日新月异 新 事物 新 观念 层出不穷 在 这样 的 背景 下 两代 人 之间 产生 一定 的 观念 分歧 是 正常 的 但 代沟 又 不是 必然 会 产生 的 同时 也 不是 不可 弥合 的 只要 两代 人 之间 相互尊重 相互理解 就 不 存在 不可逾越 的 鸿沟 言传身教 父母 的 责任 主持人 家庭 是 人生 的 第一 课堂 父母 是 人生 的 第一任 教师 从 混沌初开 的 孩童 到 学会 用 自己 眼光 看 世界 的 成人 这个 成长 的 历程 离不开 父母 的 言传身教 在 知识 和 文化教育 功能 越来越 社会化 的 情况 下 家庭教育 的 一个 主要 职能 就是 伦理道德 教育 父母 作为 子女 的 启蒙 老师 有 责任 将 一些 基本 的 伦理道德 规范 和 行为准则 传授给 子女 家庭教育 不能 停留 在 口头上 的 训导 更 重要 的 是 作 父母 的 身体力行 所谓 身教 重于 言教 陈家彬 湖北 监利 县委 组织部 在 家庭 中 对 孩子 们 从小 进行 正确 的 人生 启蒙教育 培养 其 积极 健康 的 自律意识 将 使 孩子 们 终生 受益 首先 要 帮助 孩子 正确认识 自我 这是 培养 自律意识 的 前提 在 称呼 上 父母 应多 直接 叫 孩子 的 姓名 使 孩子 形成 正确 的 自我 主体 概念 在生活上 父母 应 尽量 让 孩子 自己 保管 自己 的 衣物 鞋袜 等 日常用品 给 孩子 营造 一个 相对 固定 的 生活空间 使 孩子 形成 准确 的 自我 拥有 概念 在 行为 上 父母 应 鼓励 孩子 承担 力所能及 的 家庭事务 对 某些 经常性 劳动 如 扫地 抹 桌子 等 最好 形成 惯例 使 孩子 形成 明确 的 自我 责任 概念 其次 要 对 孩子 进行 爱心 教育 这是 培养 自律意识 的 核心 父母 应该 不仅 使 孩子 逐渐 懂得 珍惜 自己 拥有 的 一切 教会 孩子 热爱 自我 还要 告诉 孩子 有 好吃 的 要 留 几份 给 爷爷奶奶 爸爸妈妈 有 好玩 的 要 与 同伴 分享 教会 孩子 热爱 他人 爱心 教育 不仅 能 强化 孩子 的 自我意识 使 孩子 将 自我 与 环境 严格 区分 开来 还 能 帮助 孩子 进行 恰当 的 自我 定位 从而 增强 其 对 环境 的 适应性 和 创造精神 第三 要 对 孩子 严格要求 这是 培养 自律意识 的 必要 手段 对 孩子 来说 从 他律 到 自律 是 必经 过程 父母 应 切实 履行 好 教育者 的 责任 当 好孩子 的 第一任 教师 实现 家庭 的 教育 功能 当 孩子 无理 哭闹 时 父母 要 狠 得 下心 切忌 眼泪 哗哗 要 啥 给 啥 以免 孩子 形成 蛮不讲理 的 性格 当 孩子 犯错误 时 父母 要 及时 指出 严肃 批评 切不可 视而不见 听而不闻 甚至 包庇 袒护 樊天林 河南 南阳 中医药 学校 邻居 的 孩子 读 小学 有 一天 他 在 放学 的 路上 抽烟 被 我 看见 我 问 吸烟 危害 健康 你 知道 吗 他 答 知道 我 爸爸 整天 抽烟 他 不 知道 吗 儿女 沾染 坏习惯 父母 不能 只 责怪 儿女 不争气 应该 从 自己 身上 找 原因 因为 父母 的 言行 对 孩子 起着 潜移默化 的 影响 儿女 既 可以 从 父母 身上 学到 优点 又 可以 学到 缺点 儿童 早期 的 行为 特征 是 模仿 父母 是 孩子 最早 最 长期 的 模仿 对象 由于 儿童 的 年龄 小 辨别 善恶 美丑 的 能力差 因此 父母 自身 的 行为 对 儿女 来说 是 最 有力 最 形象 的 教育 为 人 父母 者 要 教育 好 自己 的 孩子 必须 从 自己 日常生活 的 一言一行 做起 主持人 从 一定 角度 讲 子女 是 父母 的 一面镜子 他们 不仅 将 父母 的 遗传 优势 忠实 地 继承 下来 同时 也 真实 地 折射出 父母 为人处世 的 哲学 和 做人 的 准则 父母 对 子女 的 示范 效应 体现 在 日常生活 的 每 一个 方面 每 一个 角落 当 子女 出现 失范 行为 时 做 父母 的 在 批评 子女 的 同时 是不是 应该 反躬自问 认真 地 检讨 一下 自己 的 行为 呢 双管齐下 根本 的 措施 主持人 子女 的 健康成长 离不开 家庭 和 社会 这 两个 环境 的 优化 和 配合 一种 清晰 的 伦理道德 规范 的 形成 需要 家庭教育 和 社会教育 的 同步 和 一致 当 家庭教育 和 社会教育 不 一致 时 必然 导致 孩子 认识 上 的 模糊 和 观念 上 的 混乱 因此 在 培养 具备 新型 伦理道德 规范 的 社会主义 接班人 过程 中 家庭教育 和 社会教育 尤其 是 学校 教育 的 密切配合 是 极其重要 的 郑 保健 中国工商银行 河南 汤阴县 支行 下班 途中 看到 一位 家长 心痛 地 抱 起 与 别人 打架 的 儿子 说 你 没长手 怎么 不 打 他 幼儿园 里 一个 小孩 追上 另 一 小朋友 狠狠 地 揍 了 他 几下 园长 很 是 吃惊 连忙 赶过去 问 那 小孩 你 怎么 这么 狠地 打 同学 呢 打人 的 小孩 回答 我 奶奶 说 了 谁 撞 了 你 你 就 狠狠 地 揍 他 园长 感慨 地说 我们 辛辛苦苦 教育 半天 顶不上 家长 的 一句 话 家庭教育 如果 与 我们 的 教育 不 同步 很难 收到 理想 的 效果 爱 子之心 人皆有之 但 爱 孩子 要会 爱 尤其 要 教会 孩子 拥有 爱心 上述 那种 爱法 并 不利于 孩子 的 健康成长 凌传昌 江西 兴国 县城 岗 乡政府 在 我 读 小学 二年级 的 那个 冬季 下 了 一场 大雪 我 的 手 被 冻得 又 红 又 肿望 着 同学 们 都 戴 着 漂亮 的 手套 我 十分 羡慕 吵 着 要 父亲 也 给 我 买 一双 父亲 当时 说 孩子 钱 不够 明天 给 你 买 第二天 父亲 没 吃 早饭 就 出门 了 到 天黑 才 回来 一进 家门 就 递给 我 一双 漂亮 的 手套 望 着 手套 我 心里 喜滋滋 的 可是 父亲 回来 后 就 病 了 一连几天 也 没能 起床 后来 我 才 知道 父亲 是 冒 着 寒冷 下河 摸 了 鱼 拿到 集市 上 卖 了 给 我 买 回 了 手套 这件 事 我 一直 记在 心中 它 鞭策 我 在 人生旅途 中 信守 自己 的 每个 诺言 主持人 一张白纸 好画 最新 最美 的 图画 子女 是 家庭 的 希望 是 国家 的 未来 家庭 和 社会 都 有 责任 将 最 纯洁 最 绚丽 的 颜色 描绘 到 这幅 图画 上 当前 要 认真 解决 家庭教育 与 社会 及 学校 教育 脱节 的 问题 在 事关 子女 前途 民族 命运 的 大 问题 上 双管齐下 密切配合 将 今天 的 孩子 培养 成 下个世纪 的 栋梁"]
    print(len(sentence))
    sentence_seq = my_token.texts_to_sequences(sentence)
    sentence_seq_pad = sequence.pad_sequences(sentence_seq, maxlen=380)

    # y = my_model.predict_classes(testBunch.contents_seq_pad[:4])
    # print(y)
    # print(type(y))
    # print(len(y))
    # print(y.shape)
