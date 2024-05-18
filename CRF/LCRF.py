import sklearn_crfsuite
from sklearn import metrics

# 定义训练数据和标签
X_train = [
    [{'token': 'I', 'pos': 'PRON'}, {'token': 'am', 'pos': 'VERB'}, {'token': 'fine', 'pos': 'ADJ'}],
    [{'token': 'How', 'pos': 'ADV'}, {'token': 'are', 'pos': 'VERB'}, {'token': 'you', 'pos': 'PRON'}]
]

y_train = [['PRON', 'VERB', 'ADJ'], ['ADV', 'VERB', 'PRON']]

# 创建 CRF 模型实例
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)

# 训练 CRF 模型
crf.fit(X_train, y_train)

# 测试数据
X_test = [[{'token': 'You', 'pos': 'PRON'}, {'token': 'are', 'pos': 'VERB'}, {'token': 'great', 'pos': 'ADJ'}]]
y_test = [['PRON', 'VERB', 'ADJ']]

# 预测标签
y_pred = crf.predict(X_test)

# 打印预测结果
print("Predicted labels:", y_pred[0])

# 评估模型性能
print(metrics.classification_report(y_test[0], y_pred[0]))

