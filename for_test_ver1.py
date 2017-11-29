import tensorflow as tf


x = tf.Variable(tf.zeros([1]), dtype=tf.float32)
cnt = 0
#delta = tf.zeros([1])
#y = []
for k in range(2):
    delta = tf.add(tf.to_float(k), tf.ones([1]))
    #y.append( tf.assign(x, delta) )
    cnt = cnt + 1
print(cnt)
#delta = tf.add(x, tf.ones([1]))
y = tf.assign(x, delta) 


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(3):
    print( sess.run( y ) )


