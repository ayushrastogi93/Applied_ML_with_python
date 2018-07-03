import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

def read_dataset():
    df = pd.read_csv("D:\ML_datasets\datasets-master\Sonar.csv")
    X = df[df.columns[0:60]].values
    Y = df[df.columns[60]].values
  #  onehotencoder = OneHotEncoder(categorical_features=[1])
   # y = onehotencoder.fit_transform[x,(y)]
    return(X,Y)
X,Y =  read_dataset()
#x,y = shuffle(x,y,randaom_state = 1) # it will be taken care during split

#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#print(sklearn.__ver)
#print(X_train.shape) #print(y_train.shape)  #print(X_test.shape)
#parameters to work with tensorflow 
learning_rate = 0.3
training_epochs=1000
cost_history = np.empty(shape=[1],dtype=float) 
mse_history = []
accuracy_history = []
n_dim = X.shape[1]
n_class = 2
model_path = 'D:\ML_datasets\Applied_ML_with_python'

n_hidden_1=60
n_hidden_2=60
n_hidden_3=60
n_hidden_4=60

x = tf.placeholder(tf.float32,[None,n_dim])
y_ = tf.placeholder(tf.float32,[None,n_class])
w = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros(n_class))

weights = {
           'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
           'h2':tf.Variable(tf.truncated_normal([n_dim,n_hidden_2])),
           'h3':tf.Variable(tf.truncated_normal([n_dim,n_hidden_3])),
           'h4':tf.Variable(tf.truncated_normal([n_dim,n_hidden_4])),
           'out':tf.Variable(tf.truncated_normal([n_class])),
                                                }
                                                
baises = {
           'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),
           'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),
           'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
           'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
           'out':tf.Variable(tf.truncated_normal([n_class])),
                                                }
def multilayer_perceptron(x,weights,baises):
    layer_1 = tf.add(tf.matmul(x,weights['h1'],baises['b1']))
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2'],baises['b2']))
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2,weights['h3'],baises['b3']))
    layer_3 = tf.nn.sigmoid(layer_3)
    layer_4 = tf.add(tf.matmul(layer_3,weights['h4'],baises['b4']))
    layer_4 = tf.nn.relu(layer_4)
    out_layer = tf.add(tf.matmul(layer_4,weights['out'],baises['out']))
    return out_layer
    
#initialize all layers 
init = tf.global_variables_initializer()
saver = tf.train.Saver() # to save model
#call the function 
y = multilayer_perceptron(x,weights,baises)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
sess = tf.session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict = {x:X_train , y:y_train})
    cost = sess.run(cost_function,feed_dict = {x:X_train , y:y_train})
    y_pred = sess.run(y,feed_dict={x:X_test})
    print ('epoch:',epoch,'-','cost:',cost)
    
save_path = saver.save(sess.model_path)
print ("model path: %s",save_path)

