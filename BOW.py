import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf

from tffm import TFFMClassifier

'''
#nltk.download()

#load file
train = pd.read_csv("../labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

#print(train.shape)
#print( train["review"][0])


tmp = bs(train["review"][0], "lxml")

#print(train["review"][0])
#print(tmp.get_text())

#去掉标点，数字，只保留单词
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      tmp.get_text() )  # The text to search
print(letters_only)


#转成全小写
lower_case = letters_only.lower()

#拆分成一个个的单词
words = lower_case.split()
print(words)

#过滤掉stopwords
words = [w for w in words if not w in stopwords.words("english")]

print(words)


'''
#dataset = pd.read_csv("amazon.txt", header=0,delimiter="\t", quoting=3)
dataset = pd.read_csv("document.tsv", header=0,delimiter="\t", quoting=3)
#dataset = pd.read_csv("training.txt", header=0,delimiter="\t", quoting=3)

#..........................
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


clean_review = review_to_words(dataset['review'][0])
print (clean_review)

# Get the number of reviews based on the dataframe column size
#num_reviews = dataset["review"].size

#num_reviews = 25000
num_reviews = 7086
#num_reviews = 1000
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews):
    # If the index is evenly divisible by 1000, print a message
    #if( (i+1)%1000 == 0 ):
        #print ("Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append( review_to_words( dataset["review"][i] ))

#print(clean_train_reviews)


'''
the below code is doing the feature extraction
'''
from sklearn.feature_extraction.text import CountVectorizer

#create a CountVectorizer object
vectorizer = CountVectorizer(analyzer="word",max_features=100)

#get the train_array_X
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
train_data_features = np.asarray(train_data_features)
#get the target
#amazon use output
#train_data_output = dataset['output']
train_data_output = dataset['sentiment']
train_data_output = np.asarray(train_data_output)
print(vectorizer.get_feature_names())

'''
the below code is FNN.....................................................................................................................
'''
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import roc_auc_score
#use those two evaluation method
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

best_auc = 0
best_auc_3d = 0

'''
init the params of FM
'''
#length of one hot encoding
one_hot_encoding_length  = 50   #1+50+50*10 we have 50 features
#length of vector
k = 16
print ('initialising')
#0.05 for amazon using bag of words
#init_weight = 0.05
#0.05 for document using bag of words
init_weight = 0.05
#uniform distribution
v = (np.random.rand(one_hot_encoding_length, k) - 0.5) * init_weight
v_3d = v
#v_3d = (np.random.rand(one_hot_encoding_length, k) - 0.5) * init_weight

#initialize weights of each bit
w = np.zeros(one_hot_encoding_length)
w_3d = np.zeros(one_hot_encoding_length)
w_0 = 0
w_0_3d = 0

weight_decay = 1E-6
learning_rate = 0.001
v_weight_decay = 1E-6
train_rounds = 7
'''
the following code is to train the FM

in total we have 50 features (key words)

['also', 'bad', 'battery', 'best', 'better', 'bluetooth', 'bought', 'buy', 'car', 'case', 'charger', 'comfortable', 'could', 'ear', 'easy', 'even', 'ever', 'excellent', 'first', 'get', 'good', 'great', 'happy', 'headset', 'item', 'like', 'love', 'money', 'new', 'nice', 'one', 'phone', 'poor', 'price', 'product', 'purchase', 'quality', 'really', 'reception', 'recommend', 'service', 'sound', 'time', 'use', 'used', 'well', 'work', 'worked', 'works', 'would']

output:
0 or 1

Class Distribution:
    --     total: 1000 instances

'''
'''
one_hot_encoding_map = {0:"buying_vhigh",1:"buying_high",2:"buying_med",3:"buying_low",4:"maint_vhigh",5:"maint_high",6:"maint_med",7:"maint_low",8:"doors_2",9:"doors_3",10:"doors_4",11:"doors_5more",12:"persons_2",13:"persons_4",14:"persons_more",15:"lug_small",16:"lug_med",17:"lug_big",18:"safe_low",19:"safe_med",20:"safe_high"}
'''
'''
read the input
'''
#read the data
#dataset_total = np.zeros((1728,7)) 1000rows *(51+1)cols
#dataset_total = np.array(train_data_features,train_data_output)
#print(train_data_features.shape)
#print(train_data_output)
dataset_total = np.zeros(shape=(num_reviews,51),dtype=int)
dataset_total = np.asarray(dataset_total)

for i in range(num_reviews):
    for j in range(one_hot_encoding_length):
        dataset_total[i][j] = train_data_features[i][j]
    dataset_total[i][one_hot_encoding_length] = train_data_output[i]

#initialize the X array and y array
'''
dataset_array_X = np.zeros((1728,6))
dataset_array_y = np.zeros((1728,1))
dataset_test_X =
dataset_test_y
'''
#iterate every line to load all data set

'''
generate training set and testing set ratio is 8:2
'''

dataset_train,dataset_test = train_test_split(dataset_total,test_size=0.3)

'''
define sigmoid train function
'''

dataset_train_y =dataset_train[:,one_hot_encoding_length]
dataset_train_X = dataset_train[:,0:one_hot_encoding_length]

#first, get the true output y and input x
dataset_test_y = dataset_test[:,one_hot_encoding_length]
dataset_test_X = dataset_test[:,0:one_hot_encoding_length]

'''
the below is for the lstm
'''

'''
the below is for the lstm
'''

'''
RF
'''
'''
the below code is RF
'''
print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
'''
random forest
'''
forest = forest.fit(dataset_train_X, dataset_train_y)
pred_y = forest.predict(dataset_test_X)

rfauc = roc_auc_score(y_true=dataset_test_y,y_score=pred_y)
rflogloss = log_loss(y_true=dataset_test_y,y_pred=pred_y)
rfmse = math.sqrt(mean_squared_error(y_true=dataset_test_y,y_pred=pred_y))

print("random forest auc is",rfauc)
print("random forest log loss is",rflogloss)
print("random forest rmse is",rfmse)

'''
naive bayes
'''
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
'''
the below is knn
'''
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
y_pred_knn = neigh.fit(X=dataset_train_X,y=dataset_train_y).predict(dataset_test_X)
knnauc = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_knn)
print ("-------------Training the knn-------------")
print("knn auc is",knnauc)


'''


'''


#exit(0)
'''
--------------------------------------------------------------
'''
def sigmoid(p):
    return  1.0 / (1.0 + math.exp(-p))
'''
define predict function in order to calculate the loss function
x represent non-zero bit for per input
'''
def pred(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    for bit in range(len(x)):
        tmp = v[int(bit)] * x[bit]# 1 needs change to 2 ,3 or more!
        sum_1 += tmp
        sum_2 += tmp * tmp
    #print("sum 1 is ",sum_1)
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    for bit in range(x.shape[0]):
        p += w[int(bit)] * x[bit]
    p = sigmoid(p)
    return (p, sum_1)


def pred_3d(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    #2d
    for bit in range(len(x)):
        tmp = v[int(bit)] * x[bit]# 1 needs change to 2 ,3 or more!
        sum_1 += tmp
        sum_2 += tmp * tmp
    #print("sum 1 is ",sum_1)
    #p = np.sum(sum_1 * sum_1)-np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0
    #p = np.sum(sum_1 * sum_1)

    for bit in range(len(x)-1):
        tmp = v_3d[int(bit)] * x[bit]# 1 needs change to 2 ,3 or more!
        tmp_1 = v_3d[int(bit)+1] * x[bit+1]# 1 needs change to 2 ,3 or more!
        sum_3 += tmp*tmp*tmp_1
    #3d
    for bit in range(len(x)):
        tmp = v_3d[int(bit)] * x[bit]# 1 needs change to 2 ,3 or more!
        sum_1 += tmp
        sum_2 += tmp * tmp * tmp
    #print("sum 1 is ",sum_1)
    #p = np.sum(sum_1 * sum_1)-np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    p += np.sum(sum_1 * sum_1 * sum_1 - sum_2 - 3 * sum_3) / 2.0
    #p = np.sum(sum_1 * sum_1)

    for bit in range(x.shape[0]):
        p += w[int(bit)] * x[bit]
    p = sigmoid(p)
    return (p, sum_1)

'''
for each line of X matrix, get the weights w and vector v. intercept is w_0
start train!
'''
#for line in dataset_array_X:


for round in range(1, train_rounds+1):
    #for training
    for line in range(dataset_train_X.shape[0]):
        #get the corresponding y
        current_y = dataset_train_y[line]
        #predict the output by current input line
        (p, vsum) = pred(dataset_train_X[line])
        #print(" line is ", line,"p is ",p)
        #(p3,vsum3)=pred_3d(dataset_train_X[line])

        d = (current_y - p)
        #d3 =(current_y - p3)
        w_0 = w_0 * (1 - weight_decay) + learning_rate * d
        #w_0_3d = w_0_3d * (1 - weight_decay) + learning_rate * d3
        #every bit in one row
        for i in range(dataset_train_X[line].shape[0]):
            w[int(i)] = w[int(i)] * (1 - weight_decay) + learning_rate * d * dataset_train_X[line][i]
            #w_3d[int(i)] = w_3d[int(i)] * (1 - weight_decay) + learning_rate * d3 * dataset_train_X[line][i]
        for j in range(dataset_train_X[line].shape[0]):
            v[int(j)] = v[int(j)] * (1 - v_weight_decay) + \
                        learning_rate * d * (dataset_train_X[line][j] * vsum - v[int(j)] * dataset_train_X[line][j] *dataset_train_X[line][j])
            #v_3d[int(j)] = v_3d[int(j)] * (1 - v_weight_decay) + learning_rate * d3 * (dataset_train_X[line][j] * vsum - v_3d[int(j)] * dataset_train_X[line][j] *dataset_train_X[line][j])

print()

'''
the code below is for testing
'''
#for testing(using f1 score),using the params we just trained to predict and get the f1 score



#get the predict y
yp = []
yp_3d = []

for row in range(dataset_test_X.shape[0]):
    (p, vsum) = pred(dataset_test_X[row])
    yp.append(p)
    #(p_3d,vsum3) = pred_3d(dataset_test_X[row])
    #yp_3d.append(p_3d)

fm_rmse = math.sqrt(mean_squared_error(y_true=dataset_test_y, y_pred=yp))
fm_auc = roc_auc_score(y_true=dataset_test_y,y_score=yp)
fm_logloss = log_loss(y_true=dataset_test_y, y_pred=yp)

print()
print("---------------train the FM---------------------")
print("fm auc is ",fm_auc)
print("fm logloss is ", fm_logloss)
#print("fm rmse is ",fm_rmse)

print("-----------------fm_3d-----------------")
#fm_rmse_3d = math.sqrt(mean_squared_error(y_true=dataset_test_y, y_pred=yp_3d))
#fm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=yp_3d)
#fm_logloss_3d = log_loss(y_true=dataset_test_y, y_pred=yp_3d)
print()
#print("fm_auc_3d is ",fm_auc_3d)
#print("fm_logloss_3d is ",fm_logloss_3d)
#print("fm_rmse_3d is ",fm_rmse_3d)
'''

from tffm import TFFMClassifier
model = TFFMClassifier(
    order=2,
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
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
    rank=one_hot_encoding_length,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)

model.fit(dataset_train_X, dataset_train_y, show_progress=True)
predict = model.predict(X=dataset_test_X)

tfm_auc_3d = roc_auc_score(y_true=dataset_test_y,y_score=predict)
print("10d fm is",tfm_auc_3d)

'''


#define function to draw confusion matrix

#exit(1)
#-----------------------------------------------------
'''
the code below is training FNN
'''
import theano
import theano.tensor as T
import linecache
import time


def get_batch_data(index, size,limit):  # 1,5->1,2,3,4,5
    array_X = []
    array_y = []
    array_F = []

    for i in range(index, index + size):
        # i: 行号,获得training set中的size个数据
        if(i==limit):
            break;

        array_f = dataset_train_X[i]
        y_tmp = dataset_train_y[i]
        x_tmp = []

        #construct the input vector for training FNN
        x_tmp.append(w_0)
        #for j in dataset_train_X[i]:
        for j in range(dataset_train_X[i].shape[0]):
            x_tmp.append(w[int(j)]*dataset_train_X[i][j])
            for tpm_k in range(k):
                x_tmp.append(v[int(j)][tpm_k]*dataset_train_X[i][j])

        array_F.append(array_f)
        array_X.append(x_tmp)
        array_y.append(int(y_tmp))
    #进行格式化
    xarray = np.array(array_X, dtype=theano.config.floatX)
    yarray = np.array(array_y, dtype=np.int32)
    return array_F,xarray, yarray


def feat_layer_one_index(feat, l):
    return 1 + int(feat)* k + l #注意feat_field，输入是feat：在十几万中的哪一位，输出是一个数字，在0到15之间

# get x array and y
def get_xy(line):
        # 说明y只有一个就是，之前的那个item
        y = int(line[0:line.index(',')])
        # x是，后面的整体
        x = line[line.index(',') + 1:]
        # 再将之后的所有按照，隔开
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
        #rmse = math.sqrt(mean_squared_error(y, yp))
        return auc
#Test Err:179	0.755578769559

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rng = np.random
rng.seed(1234)
batch_size=25                                                          #batch size
lr=0.002                                                                #learning rate
lambda1=0.01 # .01                                                        #regularisation rate
hidden1 = 600 															#hidden layer 1
hidden2 = 300															#hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 300

#initialize the w of NN
length = 851 #1+6+6*2 1+22+22*10 1+50+50*10

#uniform distribute weight matrix of first layer
w_layer1=rng.uniform(low=-np.sqrt(10. / (length + hidden1)),high=np.sqrt(10. / (length + hidden1)),size=(length,hidden1))

#activation function
if acti_type=='sigmoid':
    ww1=np.asarray((w_layer1))
elif acti_type=='tanh':
    ww1=np.asarray((w_layer1*4))
else:
    ww1=np.asarray(rng.uniform(-1,1,size=(length,hidden1)))

#intercept
bb1=np.zeros(hidden1)


#layer2的W，初始化
w_layer2=rng.uniform( low=-np.sqrt(10. / (hidden1 + hidden2)),
                high=np.sqrt(10. / (hidden1 + hidden2)),
                size=(hidden1,hidden2))
if acti_type=='sigmoid':
    ww2=np.asarray((w_layer2))
elif acti_type=='tanh':
    ww2=np.asarray((w_layer2*4))
else:
    ww2=np.asarray(rng.uniform(-1,1,size=(hidden1,hidden2)))
#截据b2
bb2=np.zeros(hidden2)
#from hidden2 to output的w
ww3=np.zeros(hidden2)


# Declare Theano symbolic variables, the interface
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
w3 = theano.shared(ww3, name="w3")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(bb2, name="b2")
b3 = theano.shared(0. , name="b3")

x_drop=dropout=0.5

# Construct Theano expression graph
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

# intercept
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

# intercept
d2 = d2 * r2[0]

p_drop = (1 / (1 + T.exp(-T.dot(d2, w3) - b3)))

#final output
p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))  # Probability that target = 1
prediction = p_1  # > 0.5                                   # The prediction thresholded
xent = - y * T.log(p_drop) - (1 - y) * T.log(1 - p_drop)  # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w3 ** 2).sum() + (b3 ** 2))  # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])  # Compute the gradient of the cost


# Compile
print("x is ",x)
print("y is ", y)

train = theano.function(
          inputs=[x,y],#x是700行551列，y是700行1列
          outputs=[gx, w1, w2, w3,b1,b2,b3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)),allow_input_downcast=True)

predict = theano.function(inputs=[x], outputs=prediction)
print("predict is ",predict)


# Train
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
        gx, w1t, w2t, w3t, b1t, b2t, b3t = train(x, y)  # gx，100行177列，177*300，300*100，300*1，100*1，1*1
        b_size = len(f)
        for t in range(b_size):
            ft = f[t]  # ft是一个数组，16位，1，6，31，45，。。。
            gxt = gx[t]  # gx,100*177,gxt 177*1
            for bit in range(len(ft)):
                if(ft[bit] != 0):
                    for l in range(k):  # k是向量的长度，11
                        v[int(bit)][l] = v[int(bit)][l] * (1 - 2. * lambda_fm * lr / b_size) \
                                                - lr * gxt[feat_layer_one_index(int(bit), l)] * ft[bit]

    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print('training: ' + str(mins) + 'm ' + str(secs) + 's')

    start_time = time.time()
    #print_err(train_file, '\t\tTraining Err: \t' + str(i))  # train error
    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print('training error: ' + str(mins) + 'm ' + str(secs) + 's')




    #test the nn using auc and rmse
    start_time = time.time()
    auc = get_err_bat()
    if auc>best_auc:
        best_auc=auc
        test_time = time.time() - start_time
        mins = int(test_time / 60)
        secs = int(test_time % 60)
        print('AUC Err:' + str(i) + '\t' + str(best_auc)+'\t')
        #print("RMSE is ",rmse)
        print('test error: ' + str(mins) + 'm ' + str(secs) + 's')
    print("best auc is", best_auc)
    # stop training when no improvement for a while
    '''
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
    '''

#-------------------------------------------------------------

'''
the code below is training FNN_3d
'''
import theano
import theano.tensor as T
import linecache
import time


def get_batch_data_3d(index, size,limit):  # 1,5->1,2,3,4,5
    array_X = []
    array_y = []
    array_F = []

    for i in range(index, index + size):
        # i: 行号,获得training set中的size个数据
        if(i==limit):
            break;

        array_f = dataset_train_X[i]
        y_tmp = dataset_train_y[i]
        x_tmp = []

        #construct the input vector for training FNN
        x_tmp.append(w_0)
        #for j in dataset_train_X[i]:
        for j in range(dataset_train_X[i].shape[0]):
            x_tmp.append(w_3d[int(j)]*dataset_train_X[i][j])
            for tpm_k in range(k):
                x_tmp.append(v_3d[int(j)][tpm_k]*dataset_train_X[i][j])

        array_F.append(array_f)
        array_X.append(x_tmp)
        array_y.append(int(y_tmp))
    #进行格式化
    xarray = np.array(array_X, dtype=theano.config.floatX)
    yarray = np.array(array_y, dtype=np.int32)
    return array_F,xarray, yarray


def feat_layer_one_index(feat, l):
    return 1 + int(feat)* k + l #注意feat_field，输入是feat：在十几万中的哪一位，输出是一个数字，在0到15之间

# get x array and y
def get_xy(line):
        # 说明y只有一个就是，之前的那个item
        y = int(line[0:line.index(',')])
        # x是，后面的整体
        x = line[line.index(',') + 1:]
        # 再将之后的所有按照，隔开
        arr = [float(xx) for xx in x.split(',')]
        return arr, y

def get_err_bat_3d():
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
                 x_tmp.append(w_3d[col]*dataset_test_X[row][col])
                 for tmp_k in range(k):
                    x_tmp.append(v_3d[col][tmp_k]*dataset_test_X[row][col])
            x_input.append(x_tmp)
        xarray = np.array(x_input, dtype=theano.config.floatX)
        pred = predict(xarray)
        for p in pred:
            yp.append(p)
        auc = roc_auc_score(dataset_test_y, yp)
        #rmse = math.sqrt(mean_squared_error(y, yp))
        return auc#,rmse
#Test Err:179	0.755578769559

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rng = np.random
rng.seed(1234)
batch_size=25                                                          #batch size
lr=0.002                                                                #learning rate
lambda1=0.01 # .01                                                        #regularisation rate
hidden1 = 600 															#hidden layer 1
hidden2 = 300															#hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 300

#initialize the w of NN
length = 851 #1+6+6*2 1+22+22*10 1+50+50*10

#uniform distribute weight matrix of first layer
w_layer1=rng.uniform(low=-np.sqrt(10. / (length + hidden1)),high=np.sqrt(10. / (length + hidden1)),size=(length,hidden1))

#activation function
if acti_type=='sigmoid':
    ww1=np.asarray((w_layer1))
elif acti_type=='tanh':
    ww1=np.asarray((w_layer1*4))
else:
    ww1=np.asarray(rng.uniform(-1,1,size=(length,hidden1)))

#intercept
bb1=np.zeros(hidden1)


#layer2的W，初始化
w_layer2=rng.uniform( low=-np.sqrt(10. / (hidden1 + hidden2)),
                high=np.sqrt(10. / (hidden1 + hidden2)),
                size=(hidden1,hidden2))
if acti_type=='sigmoid':
    ww2=np.asarray((w_layer2))
elif acti_type=='tanh':
    ww2=np.asarray((w_layer2*4))
else:
    ww2=np.asarray(rng.uniform(-1,1,size=(hidden1,hidden2)))
#截据b2
bb2=np.zeros(hidden2)
#from hidden2 to output的w
ww3=np.zeros(hidden2)


# Declare Theano symbolic variables, the interface
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
w3 = theano.shared(ww3, name="w3")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(bb2, name="b2")
b3 = theano.shared(0. , name="b3")

x_drop=dropout=0.5

# Construct Theano expression graph
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

# intercept
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

# intercept
d2 = d2 * r2[0]

p_drop = (1 / (1 + T.exp(-T.dot(d2, w3) - b3)))

#final output
p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))  # Probability that target = 1
prediction = p_1  # > 0.5                                   # The prediction thresholded
xent = - y * T.log(p_drop) - (1 - y) * T.log(1 - p_drop)  # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w3 ** 2).sum() + (b3 ** 2))  # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])  # Compute the gradient of the cost


# Compile
print("x is ",x)
print("y is ", y)

train = theano.function(
          inputs=[x,y],#x是700行551列，y是700行1列
          outputs=[gx, w1, w2, w3,b1,b2,b3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)),allow_input_downcast=True)

predict = theano.function(inputs=[x], outputs=prediction)
print("predict is ",predict)


# Train
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
        f, x, y = get_batch_data_3d(index, batch_size,limit=dataset_train_X.shape[0])
        index += batch_size
        gx, w1t, w2t, w3t, b1t, b2t, b3t = train(x, y)  # gx，100行177列，177*300，300*100，300*1，100*1，1*1
        b_size = len(f)
        for t in range(b_size):
            ft = f[t]  # ft是一个数组，16位，1，6，31，45，。。。
            gxt = gx[t]  # gx,100*177,gxt 177*1
            for bit in range(len(ft)):
                if(ft[bit] != 0):
                    for l in range(k):  # k是向量的长度，11
                        v[int(bit)][l] = v[int(bit)][l] * (1 - 2. * lambda_fm * lr / b_size) \
                                                - lr * gxt[feat_layer_one_index(int(bit), l)] * ft[bit]

    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print('training: ' + str(mins) + 'm ' + str(secs) + 's')

    start_time = time.time()
    #print_err(train_file, '\t\tTraining Err: \t' + str(i))  # train error
    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print('training error: ' + str(mins) + 'm ' + str(secs) + 's')

    #test the nn using auc and rmse
    start_time = time.time()
    auc = get_err_bat_3d()
    if auc>best_auc_3d:
        best_auc_3d=auc
        test_time = time.time() - start_time
        mins = int(test_time / 60)
        secs = int(test_time % 60)
        print('AUC Err:' + str(i) + '\t' + str(best_auc_3d)+'\t')
        #print('corresponding RMSE Err:' + str(i) + '\t' + str(rmse)+'\t')
        print('test error: ' + str(mins) + 'm ' + str(secs) + 's')
    print("best auc_3d is", best_auc_3d)

    # stop training when no improvement for a while
    '''
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
    '''
