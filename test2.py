import multiprocessing
import matplotlib.pyplot as plt
from helper import *
from sklearn.svm import LinearSVC
# def plot_result(result_dict, title):
#     x_list = range(len(result_dict.keys()))
#     feature_number = list(result_dict.keys())
#     f1_value = result_dict.values()
#     plt.figure(figsize=(16, 24))
#     plt.plot(x_list, f1_value, color='r',markerfacecolor='blue',marker='o')
#     plt.xlabel("select_features")
#     plt.xticks([])
#     plt.ylabel("f1_value")
#     plt.title(title)
#     for x,y  in zip(x_list, f1_value):
#         plt.text(x,y,(feature_number[x],y),ha="center",va="bottom")
#     plot_file = os.path.join(PLOT_FILE, title + ".jpg")
#     plt.savefig(plot_file) # 保存图片
#     plt.show()
#
#
# result_dict = {50: 0.704, 100: 0.752, 200: 0.781, 500: 0.806, 1000: 0.828, 2000: 0.84, 5000: 0.851, 10000: 0.862, 20000: 0.864, 50000: 0.862, 100000: 0.855, 200000: 0.843, 297835: 0.839}
# plot_result(result_dict, "MultinomialNB")

clf_model = LinearSVC()
print(str(clf_model))