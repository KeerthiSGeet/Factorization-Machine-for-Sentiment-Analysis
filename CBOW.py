import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review,"lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

train = pd.read_csv("filename", header=0, delimiter="\t", quoting=3)

#load the tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    index_tmp = 0
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
            index_tmp+=1
    return sentences

sentences = []

print ("Parsing sentences from training set")
tmp_index = 0
for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 50
min_word_count = 1
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling,sg=0)

model.init_sims(replace=True)

model_name = "50features_1minwords_1000context"
model.save(model_name)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):

    counter = 0.

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:

       if counter%1000 == 0:
           print ("Review %d of %d" % (counter, len(reviews)))

       try:
           reviewFeatureVecs[np.int(counter)] = makeFeatureVec(review, model, \
               num_features)
       except IndexError:
           return 0

       counter = counter + 1.
    return reviewFeatureVecs


clean_train_reviews = []
tmp = 0
for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, \
            remove_stopwords=False))
        tmp+=1

train_data_features = getAvgFeatureVecs(clean_train_reviews, model, num_features)

output = train["sentiment"].reshape(7086,1)[0:7086,:]

dataset_total = np.concatenate((train_data_features,output),axis=1)

for i in range(7086):
    for j in range(num_features):
        dataset_total[i][j] = train_data_features[i][j]
    dataset_total[i][num_features] = output[i]


dataset_train,dataset_test = train_test_split(dataset_total,test_size=0.3)

dataset_train_X = dataset_train[:,0:num_features]
dataset_train_y = dataset_train[:,num_features]

dataset_test_X = dataset_test[:,0:num_features]
dataset_test_y = dataset_test[:,num_features]

print("train shape 0 is ",train_data_features.shape[0])
print("train shape 1 is ",train_data_features.shape[1])

mymax_features = train_data_features.shape[1]

import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
best_auc = 0
one_hot_encoding_length  = num_features   #1+50+50*10 we have 50 features 1+200+200*10
k = 16
print ('initialising')
init_weight = 0.005
v = (np.random.rand(one_hot_encoding_length, k) - 0.5) * init_weight
v_3d = v
w = np.zeros(one_hot_encoding_length)
w_3d = np.zeros(one_hot_encoding_length)
w_0 = 0
w_0_3d = 0

weight_decay = 1E-6
learning_rate = 0.001
v_weight_decay = 1E-6
train_rounds = 7
print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(dataset_train_X, dataset_train_y)
pred_y = forest.predict(dataset_test_X)
rfauc = roc_auc_score(y_true=dataset_test_y,y_score=pred_y)
print("random forest auc is",rfauc)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_nb = gnb.fit(dataset_train_X, dataset_train_y).predict(dataset_test_X)
nbauc = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_nb)
print ("Training the naive bayes...")
print("Naive bayes auc is",nbauc)

def sigmoid(p):
    try:
        return  1.0 / (1.0 + math.exp(-p))
    except OverflowError:
        return 0

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
    order=2,
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
print("2d fm is",tfm_auc_3d)

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

yp = []

for row in range(dataset_test_X.shape[0]):
    (p, vsum) = pred(dataset_test_X[row])
    yp.append(p)
rmse = math.sqrt(mean_squared_error(dataset_test_y, yp))
fmauc = roc_auc_score(dataset_test_y,yp)
print("rmse is ",rmse)
print("fmauc is ",fmauc)

import theano
import theano.tensor as T
import linecache
import time


def get_batch_data(index, size,limit):  # 1,5->1,2,3,4,5
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
    return 1 + int(feat)* k + l #

def get_xy(line):

        y = int(line[0:line.index(',')])

        x = line[line.index(',') + 1:]

        arr = [float(xx) for xx in x.split(',')]
        return arr, y

def get_err_bat():
        yp = []
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
hidden1 = 600 															#hidden layer 1
hidden2 = 300															#hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 50
length = 151
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
    h1 = 1 / (1 + T.exp(-z1))  # hidden layer 1
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
prediction = p_1  # > 0.5
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

    if auc > min_err:
        best_w1 = w1t
        best_w2 = w2t
        best_w3 = w3t
        best_b1 = b1t
        best_b2 = b2t
        best_b3 = b3t
        min_err = auc
        min_err_epoch = i
        if times_reduce < 3:
            times_reduce += 1
    else:
        times_reduce -= 1
    if times_reduce < 0:
        break
