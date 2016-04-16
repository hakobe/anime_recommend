import numpy as np
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as OS
import chainer.optimizer as O
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

FEATURE_FILE = 'feature.tsv'
ANSWER_FILE = 'answer.tsv'
TARGET_FILE = 'target.tsv'

def over_sample(x,y):
    class_xs = []
    min_elems = None

    elems_0 = x[(y == 0)]
    elems_1 = x[(y == 1)]

    diff = elems_0.shape[0] - elems_1.shape[0]
    while elems_1.shape[0] < diff:
        elems_1 = np.concatenate([elems_1, elems_1])

    append_X = elems_1[:diff]
    append_y = np.empty(diff)
    append_y.fill(1)

    xs = np.concatenate((x, append_X))
    ys = np.concatenate((y, append_y))

    return shuffle(xs, ys)

def load_data():
    X_all = np.loadtxt(FEATURE_FILE, delimiter="\t", dtype='float32')
    y_all = np.loadtxt(ANSWER_FILE, delimiter="\t", dtype='int')
    target = np.loadtxt(TARGET_FILE, delimiter="\t", dtype='float32')

    X_train, X_rest, y_train, y_rest = cross_validation.train_test_split(X_all, y_all, test_size=0.4, random_state=5)
    X_val, X_test, y_val, y_test = cross_validation.train_test_split(X_rest, y_rest, test_size=0.5, random_state=5)

    return (X_train, y_train, X_val, y_val, X_test, y_test, target)

class AnimeChain(Chain):

    def __init__(self, n, m):
        super(AnimeChain, self).__init__(
            l1=L.Linear(n, m),
            l2=L.Linear(m, 2)
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        o = self.l2(h1)
        return o

def train(X, t, hidden_n, weight_decay):
    print(hidden_n, weight_decay)
    model = AnimeChain(X.shape[1], hidden_n)
    optimizer = OS.Adam()
    optimizer.setup(model)
    if weight_decay:
        optimizer.add_hook(O.WeightDecay(weight_decay))

    X_o, t_o = over_sample(X, t)

    for e in range(1500):
        V_X = Variable(X_o)
        V_t = Variable(np.array(t_o, dtype='int32'))

        V_y = model(V_X)
        model.zerograds()
        loss = F.softmax_cross_entropy(V_y, V_t)
        loss.backward()
        optimizer.update()
        #print(e, loss.data)
        if loss.data < 0.001:
            break
    return model

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, target = load_data()
    print(X_train.shape)

    weight_decays = [None, 0.0005]
    hidden_ns = [100, 300, 600, 1000]
    scores = []

    for weight_decay in weight_decays:
        for hidden_n in hidden_ns:
            model = train(X_train, y_train, hidden_n = hidden_n, weight_decay = weight_decay)

            last_y = model(Variable(X_val))
            yd = last_y.data
            y_cmp = (yd[:, 0] < yd[:, 1]).astype(np.int32)
            precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_cmp, beta=1, average='binary')
            print(fscore, precision, recall)
            print(confusion_matrix(y_val, y_cmp))
            scores.append([hidden_n, weight_decay, fscore, precision, recall])

    scores = sorted(scores, key=lambda x: -x[2])


    model = train(X_train, y_train, hidden_n = scores[0][0], weight_decay = scores[0][1])
    yd = model(Variable(X_val)).data
    ycmp = (yd[:, 0] < yd[:, 1]).astype(np.int32)
    print(classification_report(ycmp, y_val))

    tids_target = target[:, 0]
    X_target = target[:, 1:]
    yd = model(Variable(X_target)).data
    y_target = (yd[:, 0] < yd[:, 1]).astype(np.int32)

    for i in range(len(y_target)):
        if y_target[i] == 1:
            print("http://cal.syoboi.jp/tid/%d" % int(tids_target[i]))

main()
