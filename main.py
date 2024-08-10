from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt 

califonia=datasets.fetch_california_housing()

#features and labels 

X=califonia.data
Y=califonia.target

print(X,Y)

#algorithm
l_reg= linear_model.LinearRegression()

plt.scatter(X.T[1],Y)
plt.show()

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)
#train

model=l_reg.fit(X_train,Y_train)

predictions=model.predict(X_test)


print("predictions:", predictions)
print('R^2', l_reg.score(X,Y))
print('coeff', l_reg.coef_)
print('intercept:', l_reg.intercept_)
plt.show()