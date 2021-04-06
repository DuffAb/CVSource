# -*- coding: utf-8 -*-
#
#%%
from sklearn import linear_model
X = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]

y = [0,
     1,
     1,
     1,
     0,
     0]

lr = linear_model.LogisticRegression()
lr.fit(X, y)

testX = [[28, 8]]

label = lr.predict(testX)
print("predicted label = ", label)

prob = lr.predict_proba(testX)
print("probability = ", prob)



#%%
theta_0 = lr.intercept_
theta_1 = lr.coef_[0][0]
theta_2 = lr.coef_[0][1]

print("theta_0 = ", theta_0)
print("theta_1 = ", theta_1)
print("theta_2 = ", theta_2)

testX = [[28, 8]]

ratio = prob[0][1]/prob[0][0]


testX = [[28, 9]]
prob_new = lr.predict_proba(testX)
ratio_new = prob_new[0][1]/prob_new[0][0]

ratio_of_ratio = ratio_new / ratio
print("ratio_of_ratio = ", ratio_of_ratio)

import  math
theta2_e = math.exp(theta_2)
print("theta 2 e = ", theta2_e)
