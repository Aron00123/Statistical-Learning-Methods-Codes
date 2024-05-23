import sklearn_crfsuite
from sklearn_crfsuite import metrics

# 示例数据
train_sents = [
    [('I', 'PRP'), ('love', 'VBP'), ('coding', 'VBG')],
    [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN')]
]

test_sents = [
    [('I', 'PRP'), ('hate', 'VBP'), ('bugs', 'NNS')],
    [('Testing', 'VBG'), ('is', 'VBZ'), ('fun', 'NN')]
]

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sent) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if i == 0 else sent[i - 1][0],
        'next_word': '' if i == len(sent) - 1 else sent[i + 1][0],
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# 提取训练数据的特征和标签
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

# 创建CRF模型并训练
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

# 提取测试数据的特征和标签
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# 进行预测
y_pred = crf.predict(X_test)

# 计算模型性能
f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1_score}')

# 查看分类报告
# report = metrics.flat_classification_report(y_test, y_pred, digits=3)
# print(report)