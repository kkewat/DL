import tensorflow as tf

mat_A = tf.constant([1,2,3,4,5,6],shape=(2,3))
mat_B = tf.constant([10,20,30,40,50,60],shape=(3,2))

print("Matrix A \n\n\n",mat_A)
print("Matrix B \n\n\n",mat_B)

C = tf.matmul(mat_A,mat_B)
print("Matrix Multplication\n",C)

e_mat_A = tf.random.uniform([2,2],minval = 3, maxval=100,dtype=tf.float32,name="MatrixA")
print("\n\nMatrixA {}\n".format(e_mat_A))

eighen_value, eighen_vector = tf.linalg.eigh(e_mat_A)

print("\neighen Vector :{}\n\neighen Value: {}".format(eighen_vector,eighen_value))
