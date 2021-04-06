# -*- coding: utf-8 -*-
#%%

# 1.数据集介绍
# SMSSpamCollection.txt数据集
# 第一列是短信的label
# ham：非垃圾短信
# spam：垃圾短信
# \t键后面是短信的正文

# 2.导入要用的包
import pandas as pd
from sklearn import linear_model #引入机器学习的线性模型
from sklearn.feature_extraction.text import TfidfVectorizer # sklearn包中，特殊提取中的文本模块中，特殊字符向量化方法

# 3.读入数据集
path = './'
filename = 'SMSSpamCollection.txt'
df = pd.read_csv(path + filename, delimiter='\t', header=None) # 用\t分割，没有文件头
# 生成label和x输入
y, X_train = df[0], df[1]

# 4.预处理
# zip个数据集是文本的，因此要对label和x都做预处理
# sklearn.feature_extraction.text.TfidfVectorizer，计算单词在所给数据集中的频率和在当前数据中的频率。
# 对单词在所有数据集中的概率做导数向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)

# 5.训练模型
# 使用Logic回归方法训练
lr = linear_model.LogisticRegression()
lr.fit(X, y)

# 6.进行测试
testX = vectorizer.transform(['Urgent: hello, Your mobile was awarded a Prize!',
                             'Hello, how are you'])
predictions = lr.predict(testX)
print(predictions)
