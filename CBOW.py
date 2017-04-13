import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review,"lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

#------------------------load data---------------------------------
#load train data
#train = pd.read_csv("amazon.txt", header=0, delimiter="\t", quoting=3)
#train = pd.read_csv("document.tsv", header=0, delimiter="\t", quoting=3)
train = pd.read_csv("training.txt", header=0, delimiter="\t", quoting=3)

#load unlabeled train
#unlabel_train = pd.read_csv("unlabeledTrainData.tsv", header=0,delimiter="\t", quoting=3 )

#load test data
#test=pd.read_csv("amazon.txt", header=0, delimiter="\t", quoting=3)

#------------feature extract-------------------------------------------------

#load the tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    index_tmp = 0
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it (limit the length =1000)
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
            index_tmp+=1

    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

# convert each sentense to vectors
sentences = []  # Initialize an empty list of sentences

#deal with labeled train dataset
print ("Parsing sentences from training set")
tmp_index = 0
for review in train["review"]:
    #if tmp_index<1000:
        sentences += review_to_sentences(review, tokenizer)
    #tmp_index+=1;

#deal with unlabeled train dataset
#print ("Parsing sentences from unlabeled set")
#for review in unlabel_train["review"]:
#    sentences += review_to_sentences(review, tokenizer)

# feature extraction
#-----------------------------------------------------------------------------------------------------------------
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 50    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling,sg=0)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "50features_1minwords_1000context"
model.save(model_name)

#from sklearn.cluster import KMeans
#import time



#-------------------create vector-----------------------------------
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print ("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       try:
           reviewFeatureVecs[np.int(counter)] = makeFeatureVec(review, model, \
               num_features)
       except IndexError:
           return 0
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
tmp = 0
for review in train["review"]:
    #if tmp<25000:
        clean_train_reviews.append(review_to_wordlist(review, \
            remove_stopwords=False))
        tmp+=1


#在此处ramdon_split
train_data_features = getAvgFeatureVecs(clean_train_reviews, model, num_features)



#output = train["output"].reshape(1000,1)
#0到1000行的所有列

#output = train["sentiment"].reshape(25000,1)[0:25000,:]
output = train["sentiment"].reshape(7086,1)[0:7086,:]

dataset_total = np.concatenate((train_data_features,output),axis=1)

#dataset_total = np.zeros(shape=(1000,551),dtype=int)
#dataset_total = np.asarray(dataset_total)

#for i in range(25000):
#for i in range(1000):
for i in range(7086):
    for j in range(num_features):
        dataset_total[i][j] = train_data_features[i][j]
    dataset_total[i][num_features] = output[i]


dataset_train,dataset_test = train_test_split(dataset_total,test_size=0.3)

#from 0 to 300(excluded)
dataset_train_X = dataset_train[:,0:num_features]
#the 300th column
dataset_train_y = dataset_train[:,num_features]


#first, get the true output y and input x
dataset_test_X = dataset_test[:,0:num_features]
dataset_test_y = dataset_test[:,num_features]


#test_data = train_data_features
'''
print ("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))
'''

#testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

print("train shape 0 is ",train_data_features.shape[0])
print("train shape 1 is ",train_data_features.shape[1])

#print("test shape 0 is ",test_data.shape[0])
#print("test shape 1 is ",test_data.shape[1])

#exit(1)
'''
--------------FM------------------------
'''

'''
the below code is doing the feature extraction
'''
#mymax_features = 300
mymax_features = train_data_features.shape[1]

'''
the below code is FNN.....................................................................................................................
'''
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
best_auc = 0
'''
init the params of FM
'''
#length of one hot encoding
one_hot_encoding_length  = num_features   #1+50+50*10 we have 50 features 1+200+200*10

#length of vector
k = 2
print ('initialising')
init_weight = 0.005
#uniform distribution
v = (np.random.rand(one_hot_encoding_length, k) - 0.5) * init_weight
v_3d = v

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
#dataset_total = np.zeros((1728,7)) #1000rows *(51+1)cols
#dataset_total = np.array(train_data_features,train_data_output)
#print(train_data_features.shape)
#print(train_data_output)
#dataset_total = np.zeros(shape=(train_data_features.shape[0],one_hot_encoding_length),dtype=int)
#dataset_total = np.asarray(train_data_features)


#initialize the X array and y array
'''
dataset_array_X = np.zeros((1728,6))
dataset_array_y = np.zeros((1728,1))
dataset_test_X =
dataset_test_y
'''
#iterate every line to load all data set

'''
generate training set and testing set ratio is 7:3
define sigmoid train function
'''
#dataset_train_y = train["sentiment"]
#dataset_train_X = train_data_features

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
print("random forest auc is",rfauc)
'''
naive bayes
'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_nb = gnb.fit(dataset_train_X, dataset_train_y).predict(dataset_test_X)
nbauc = roc_auc_score(y_true=dataset_test_y,y_score=y_pred_nb)
print ("Training the naive bayes...")
print("Naive bayes auc is",nbauc)
'''
--------------------------------------------------------------
'''
def sigmoid(p):
    try:
        return  1.0 / (1.0 + math.exp(-p))
    except OverflowError:
        return 0
'''
define predict function in order to calculate the loss function
x represent non-zero bit for per input
'''

'''
the below is code of svm classification
kernal=rbf
'''

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
移走了
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
    #p = sigmoid(p)
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
        #if (line == 120):
            #print()
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
            v[int(j)] = v[int(j)] * (1 - v_weight_decay) + learning_rate * d * (dataset_train_X[line][j] * vsum - v[int(j)] * dataset_train_X[line][j] *dataset_train_X[line][j])
            #v_3d[int(j)] = v_3d[int(j)] * (1 - v_weight_decay) + learning_rate * d3 * (dataset_train_X[line][j] * vsum - v_3d[int(j)] * dataset_train_X[line][j] *dataset_train_X[line][j])

print()

'''
the code below is for testing
'''
#for testing(using f1 score),using the params we just trained to predict and get the f1 score



#get the predict y
yp = []
#yp_3d = []

for row in range(dataset_test_X.shape[0]):
    (p, vsum) = pred(dataset_test_X[row])
    #(p_3d, vsum3) = pred_3d(dataset_test_X[row])
    yp.append(p)
    #yp_3d.append(p_3d)
#calculate the f1 score and rmse and evaluation
'''
auc not work!

empirical error

'''
#auc = roc_auc_score(dataset_test_y, yp)
rmse = math.sqrt(mean_squared_error(dataset_test_y, yp))
fmauc = roc_auc_score(dataset_test_y,yp)


#rmse_3d = math.sqrt(mean_squared_error(dataset_test_y, yp_3d))
#fmauc_3d = roc_auc_score(dataset_test_y,yp_3d)

#print("f1 is ",f1)
print("rmse is ",rmse)

#define function to draw confusion matrix
print("fmauc is ",fmauc)


#print("f1 is ",f1)
#print("rmse_3d is ",rmse_3d)
#define function to draw confusion matrix
#print("fmauc_3d is ",fmauc_3d)
#-----------------------------------------------------
'''
the code below is training FNN
'''
'''
import theano
import theano.tensor as T
import linecache
import time

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rng = np.random
rng.seed(1234)
batch_size=25                                                          #batch size
lr=0.002                                                                #learning rate
lambda1=0.01 # .01                                                        #regularisation rate
hidden1 = 600 															#hidden layer 1
hidden2 = 300															#hidden layer 2
acti_type='linear'                                                    #activation type
epoch = 100

#initialize the w of NN
length = 551 #1+6+6*2 1+22+22*10 1+200+200*10

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


    #test the nn using auc and rmse
    start_time = time.time()
    auc = get_err_bat()
    if auc>best_auc:
        best_auc=auc
        test_time = time.time() - start_time
        mins = int(test_time / 60)
        secs = int(test_time % 60)
        print('Test Err:' + str(i) + '\t' + str(best_auc)+'\t')
        print('test error: ' + str(mins) + 'm ' + str(secs) + 's')
    print("best auc is", best_auc)
    # stop training when no improvement for a while




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
'''
#the code below is training FNN
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
length = 151 #1+6+6*2 1+22+22*10 1+50+50*10

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
    '''

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
length = 1+num_features*11 #1+6+6*2 1+22+22*10 1+50+50*10

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
best_auc_3d=0

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
