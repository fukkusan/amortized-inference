import tensorflow as tf


x = tf.Variable(tf.ones([1]), dtype=tf.float32)
cnt = 0
f = tf.multiply(x, x)
grad_f = []
for s in range(4):
    grad_f.append( tf.gradients(f, x) )
    cnt = cnt + 1
print(cnt)
delta = tf.add(x, tf.ones([1]))
x_prime = tf.assign(x, delta) 


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(10):
    print( sess.run( [x_prime, grad_f] ) )


