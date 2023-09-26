from sklearn.svm import SVR

x = [[0, 0, 0], [1, 1, 1]]               
y = [0, 1]

# SVR 모델 선언 후 Fitting
svr = SVR()
svr.fit(x, y)

# Fitting된 모델로 x_valid를 통해 예측을 진행
ret = svr.predict([[1, 1, 0]])

print(ret)