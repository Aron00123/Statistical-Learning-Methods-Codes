from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 创建一个示例数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化AdaBoost分类器
adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# 在训练集上拟合模型
adaboost_clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = adaboost_clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
