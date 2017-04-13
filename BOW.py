import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf

from tffm import TFFMClassifier

dataset = pd.read_csv("here input the file name", header=0,delimiter="\t", quoting=3)

def review_to_words( raw_review ):

    review_text = BeautifulSoup(raw_review,"lxml").get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return( " ".join( meaningful_words ))


clean_review = review_to_words(dataset['review'][0])
print (clean_review)

num_reviews = 7086
clean_train_reviews = []
print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews):
    clean_train_reviews.append( review_to_words( dataset["review"][i] ))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word",max_features=100)

train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
train_data_features = np.asarray(train_data_features)

train_data_output = dataset['output']
train_data_output = dataset['sentiment']
train_data_output = np.asarray(train_data_output)

import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

best_auc = 0
one_hot_encoding_length  = 50
k = 16
print ('initialising')
init_weight = 0.05
v = (np.random.rand(one_hot_encoding_length, k) - 0.5) * init_weight
v_3d = v

w = np.zeros(one_hot_encoding_length)
w_0 = 0

weight_decay = 1E-6
learning_rate = 0.001
v_weight_decay = 1E-6
train_rounds = 7

dataset_total = np.zeros(shape=(num_reviews,51),dtype=int)
dataset_total = np.asarray(dataset_total)

for i in range(num_reviews):
    for j in range(one_hot_encoding_length):
        dataset_total[i][j] = train_data_features[i][j]
    dataset_total[i][one_hot_encoding_length] = train_data_output[i]

dataset_train,dataset_test = train_test_split(dataset_total,test_size=0.3)

dataset_train_y =dataset_train[:,one_hot_encoding_length]
dataset_train_X = dataset_train[:,0:one_hot_encoding_length]

dataset_test_y = dataset_test[:,one_hot_encoding_length]
dataset_test_X = dataset_test[:,0:one_hot_encoding_length]

print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(dataset_train_X, dataset_train_y)
pred_y = forest.predict(dataset_test_X)

rfauc = roc_auc_score(y_true=dataset_test_y,y_score=pred_y)
rflogloss = log_loss(y_true=dataset_test_y,y_pred=pred_y)
rfmse = math.sqrt(mean_squared_error(y_true=dataset_test_y,y_pred=pred_y))
print("random forest auc is",rfauc)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_nb = gnb.fit(dataset_train_X, dataset_train_y).predict(dataset_test_X)
nbauc = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_nb)
nblogloss = log_loss(y_true=dataset_test_y,y_pred=y_pred_nb)
nbrmse = math.sqrt(mean_squared_error(y_true=dataset_test_y,y_pred=y_pred_nb))
print()
print ("-------------Training the naive bayes-------------")
print("Naive bayes auc is",nbauc)
print("Naive bayes log loss is",nblogloss)
print("Naive bayes rmse is",nbrmse)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
y_pred_knn = neigh.fit(X=dataset_train_X,y=dataset_train_y).predict(dataset_test_X)
knnauc = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_knn)
print ("-------------Training the knn-------------")
print("knn auc is",knnauc)

from sklearn import svm
clf = svm.SVC()
y_pred_svm = clf.fit(X=dataset_train_X,y=dataset_train_y).predict(dataset_test_X)
svmauc = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_svm)
print ("-------------Training the rbfSVM-------------")
print("rbf SVM auc is",svmauc)

from sklearn import svm
clf2 = svm.LinearSVC()
y_pred_svm2 = clf2.fit(X=dataset_train_X,y=dataset_train_y).predict(dataset_test_X)
svmauc2 = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_svm2)
print ("-------------Training the linear SVM-------------")
print("linear SVM auc is",svmauc2)

from sklearn import svm
clf3 = svm.NuSVC(kernel='sigmoid')
y_pred_svm3 = clf3.fit(X=dataset_train_X,y=dataset_train_y).predict(dataset_test_X)
svmauc3 = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_svm3)
print ("-------------Training the sigmoid SVM-------------")
print("sigmoid SVM auc is",svmauc3)

from sklearn import svm
clf4 = svm.NuSVC(kernel='poly')
y_pred_svm4 = clf4.fit(X=dataset_train_X,y=dataset_train_y).predict(dataset_test_X)
svmauc4 = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_svm4)
print ("-------------Training the poly SVM-------------")
print("poly SVM auc is",svmauc4)

from tffm import TFFMClassifier
model = TFFMClassifier(
    order=3,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("3d fm is",tfm_auc_3d)


from tffm import TFFMClassifier
model = TFFMClassifier(
    order=4,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("4d fm is",tfm_auc_3d)

from tffm import TFFMClassifier
model = TFFMClassifier(
    order=5,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("5d fm is",tfm_auc_3d)

from tffm import TFFMClassifier
model = TFFMClassifier(
    order=6,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("6d fm is",tfm_auc_3d)

from tffm import TFFMClassifier
model = TFFMClassifier(
    order=7,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("7d fm is",tfm_auc_3d)


from tffm import TFFMClassifier
model = TFFMClassifier(
    order=8,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("8d fm is",tfm_auc_3d)


from tffm import TFFMClassifier
model = TFFMClassifier(
    order=9,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("9d fm is",tfm_auc_3d)


from tffm import TFFMClassifier
model = TFFMClassifier(
    order=10,
    rank=16,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=200,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("10d fm is",tfm_auc_3d)

def sigmoid(p):
    return  1.0 / (1.0 + math.exp(-p))

def pred(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    for bit in range(len(x)):
        tmp = v[int(bit)] * x[bit]
        sum_1 += tmp
        sum_2 += tmp * tmp

    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    for bit in range(x.shape[0]):
        p += w[int(bit)] * x[bit]
    p = sigmoid(p)
    return (p, sum_1)

for round in range(1, train_rounds+1):
    for line in range(dataset_train_X.shape[0]):
        current_y = dataset_train_y[line]
        (p, vsum) = pred(dataset_train_X[line])

        d = (current_y - p)
        w_0 = w_0 * (1 - weight_decay) + learning_rate * d
        for i in range(dataset_train_X[line].shape[0]):
            w[int(i)] = w[int(i)] * (1 - weight_decay) + learning_rate * d * dataset_train_X[line][i]
        for j in range(dataset_train_X[line].shape[0]):
            v[int(j)] = v[int(j)] * (1 - v_weight_decay) + \
                        learning_rate * d * (dataset_train_X[line][j] * vsum - v[int(j)] * dataset_train_X[line][j] *dataset_train_X[line][j])

yp = []
yp_3d = []

for row in range(dataset_test_X.shape[0]):
    (p, vsum) = pred(dataset_test_X[row])
    yp.append(p)

fm_rmse = math.sqrt(mean_squared_error(y_true=dataset_test_y, y_pred=yp))
fm_auc = roc_auc_score(y_true=dataset_test_y,y_score=yp)
fm_logloss = log_loss(y_true=dataset_test_y, y_pred=yp)

print()
print("---------------train the FM---------------------")
print("fm auc is ",fm_auc)
print("fm logloss is ", fm_logloss)

import theano
import theano.tensor as T
import linecache
import time


def get_batch_data(index, size,limit):
    array_X = []
    array_y = []
    array_F = []

    for i in range(index, index + size):
        if(i==limit):
            break;

        array_f = dataset_train_X[i]
        y_tmp = dataset_train_y[i]
        x_tmp = []


        x_tmp.append(w_0)
        for j in range(dataset_train_X[i].shape[0]):
            x_tmp.append(w[int(j)]*dataset_train_X[i][j])
            for tpm_k in range(k):
                x_tmp.append(v[int(j)][tpm_k]*dataset_train_X[i][j])

        array_F.append(array_f)
        array_X.append(x_tmp)
        array_y.append(int(y_tmp))
    xarray = np.array(array_X, dtype=theano.config.floatX)
    yarray = np.array(array_y, dtype=np.int32)
    return array_F,xarray, yarray

def feat_layer_one_index(feat, l):
    return 1 + int(feat)* k + l

def get_xy(line):
        y = int(line[0:line.index(',')])
        x = line[line.index(',') + 1:]
        arr = [float(xx) for xx in x.split(',')]
        return arr, y

def get_err_bat():
        y = []
        yp = []
        flag_start = 0
        xx_bat = []
        flag = False
        x_input = []
        for row in range(dataset_test_X.shape[0]):
            x_tmp = []
            x_tmp.append(w_0)
            for col in range(dataset_test_X.shape[1]):
                 x_tmp.append(w[col]*dataset_test_X[row][col])
                 for tmp_k in range(k):
                    x_tmp.append(v[col][tmp_k]*dataset_test_X[row][col])
            x_input.append(x_tmp)
        xarray = np.array(x_input, dtype=theano.config.floatX)
        pred = predict(xarray)
        for p in pred:
            yp.append(p)
        auc = roc_auc_score(dataset_test_y, yp)
        return auc

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rng = np.random
rng.seed(1234)
batch_size=25
lr=0.002
lambda1=0.01
hidden1 = 600
hidden2 = 300
acti_type='tanh'
epoch = 50

length = 851

w_layer1=rng.uniform(low=-np.sqrt(10. / (length + hidden1)),high=np.sqrt(10. / (length + hidden1)),size=(length,hidden1))

if acti_type=='sigmoid':
    ww1=np.asarray((w_layer1))
elif acti_type=='tanh':
    ww1=np.asarray((w_layer1*4))
else:
    ww1=np.asarray(rng.uniform(-1,1,size=(length,hidden1)))

bb1=np.zeros(hidden1)

w_layer2=rng.uniform( low=-np.sqrt(10. / (hidden1 + hidden2)),
                high=np.sqrt(10. / (hidden1 + hidden2)),
                size=(hidden1,hidden2))
if acti_type=='sigmoid':
    ww2=np.asarray((w_layer2))
elif acti_type=='tanh':
    ww2=np.asarray((w_layer2*4))
else:
    ww2=np.asarray(rng.uniform(-1,1,size=(hidden1,hidden2)))
bb2=np.zeros(hidden2)
ww3=np.zeros(hidden2)

x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
w3 = theano.shared(ww3, name="w3")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(bb2, name="b2")
b3 = theano.shared(0. , name="b3")

x_drop=dropout=0.5

r0 = srng.binomial(size=(1, length), n=1, p=x_drop)
print("r0 is ", r0)
x = x * r0[0]

z1 = T.dot(x, w1) + b1
if acti_type == 'sigmoid':
    h1 = 1 / (1 + T.exp(-z1))
elif acti_type == 'linear':
    h1 = z1
elif acti_type == 'tanh':
    h1 = T.tanh(z1)
r1 = srng.binomial(size=(1, hidden1), n=1, p=dropout)
d1 = h1 * r1[0]
z2 = T.dot(h1, w2) + b2
if acti_type == 'sigmoid':
    h2 = 1 / (1 + T.exp(-z2))  # hidden layer 2
elif acti_type == 'linear':
    h2 = z2
elif acti_type == 'tanh':
    h2 = T.tanh(z2)
d2 = T.tanh(T.dot(h1, w2) + b2)
r2 = srng.binomial(size=(1, hidden2), n=1, p=dropout)
d2 = d2 * r2[0]
p_drop = (1 / (1 + T.exp(-T.dot(d2, w3) - b3)))

p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))
prediction = p_1
xent = - y * T.log(p_drop) - (1 - y) * T.log(1 - p_drop)
cost = xent.sum() + lambda1 * ((w3 ** 2).sum() + (b3 ** 2))
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])
print("x is ",x)
print("y is ", y)
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1, w2, w3,b1,b2,b3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)),allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=prediction)
print("predict is ",predict)
print("Training model:")
best_w1 = w1.get_value()
best_w2 = w2.get_value()
best_w3 = w1.get_value()
best_b1 = b1.get_value()
best_b2 = b2.get_value()
best_b3 = b3.get_value()
min_err = 0
min_err_epoch = 0
times_reduce = 100
n_batch = 50
train_size =500
lambda_fm = 0.01
for i in range(epoch):
    start_time = time.time()
    index = 0
    print(n_batch)
    for j in range(int(n_batch)):
        if index > train_size:
            break
        f, x, y = get_batch_data(index, batch_size,limit=dataset_train_X.shape[0])
        index += batch_size
        gx, w1t, w2t, w3t, b1t, b2t, b3t = train(x, y)
        b_size = len(f)
        for t in range(b_size):
            ft = f[t]
            gxt = gx[t]
            for bit in range(len(ft)):
                if(ft[bit] != 0):
                    for l in range(k):
                        v[int(bit)][l] = v[int(bit)][l] * (1 - 2. * lambda_fm * lr / b_size) \
                                                - lr * gxt[feat_layer_one_index(int(bit), l)] * ft[bit]
    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print('training: ' + str(mins) + 'm ' + str(secs) + 's')
    start_time = time.time()
    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print('training error: ' + str(mins) + 'm ' + str(secs) + 's')
    start_time = time.time()
    auc = get_err_bat()
    if auc>best_auc:
        best_auc=auc
        test_time = time.time() - start_time
        mins = int(test_time / 60)
        secs = int(test_time % 60)
        print('AUC Err:' + str(i) + '\t' + str(best_auc)+'\t')
        print('test error: ' + str(mins) + 'm ' + str(secs) + 's')
    print("best auc is", best_auc)
