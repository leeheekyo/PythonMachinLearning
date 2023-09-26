from sklearn import tree

x = [[0, 0, 0], [1, 1, 1]]               
y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
ret = clf.predict([[1, 1, 1]])

print(ret)