import tensorflow as tf
#constant variables 
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0,tf.float32)
print(node1,node2)
sess = tf.Session()
print(sess.run([node1,node2]))
sess.close()
#2 placeholders are used when we need to fetch values during runtime 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
node3 = a + b
sess = tf.Session()
#print(sess.run(node3,{a:[0.5],b:[0.6]}))
print(sess.run(node3,{a:[1,3],b:[0,6]}))
sess.close()
#3 tf.Variable we use to train our graph.models  
#model parameters
w = tf.Variable([0.3],tf.float32)
c = tf.Variable([-0.3],tf.float32)
#inouts and outputs
x = tf.placeholder(tf.float32)
linear_model = w*x + c
y = tf.placeholder(tf.float32)
#loss calculation to check efficiency of our models 
squared_deltas= tf.square(linear_model -y)
loss = tf.reduce_sum(squared_deltas)
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
 #add an operation to graph     
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)#will run te operation 
#print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
#sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
for i in range(1000):
  sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([w,c]))
sess.close()