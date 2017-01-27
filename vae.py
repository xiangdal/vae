import tensorflow as tf
import numpy as np
import datetime
import math
import cPickle as pickle
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

ver = tf.__version__
print("Tensor Flow version {}".format(ver))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
print("Number of samples {} Shape of y[{}] Shape of X[{}]"
      .format(n_samples, mnist.train.labels.shape, mnist.train.images.shape))

class VAE:
  # create weights W
  def get_W(self, shape, name=None, std=0.1):
    initial_normal = tf.truncated_normal(shape, stddev=std)
    low = -4*np.sqrt(6.0/(shape[0] + shape[1])) # use 4 for sigmoid, 1 for tanh activation 
    high = 4*np.sqrt(6.0/(shape[0] + shape[1]))
    initial_xavier = tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)
    return tf.Variable(initial_normal, name=name)

  # create bias b
  def get_b(self, shape, name=None, init=0.0):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial, name=name)

  def __init__(self, encoder_arch, decoder_arch, latent_dim, ln_rate, batch_size, sess, activation_fun=tf.nn.relu, restore=False):
    self.x = tf.placeholder("float", shape=[None, 28*28])
    self.y = tf.placeholder(tf.float32, [None, 10])
    self.encoder_arch = encoder_arch
    self.decoder_arch = decoder_arch
    self.latent_dim = latent_dim
    self.ln_rate = ln_rate
    self.batch_size = batch_size
    self.activation_fun = activation_fun
    self.sess = sess
    self.restore = restore

    self.log_name = datetime.datetime.now().strftime("%I%M%p%B%d%Y")
    
    self.decoder_arch.insert(0, self.latent_dim) # for simplicity of creating decoders
    
    with tf.name_scope('encoder'):
      self.create_encoder()

    with tf.name_scope('classifier'):
      self.create_classifier()
    
    with tf.name_scope('latent_variables'):
      self.create_latent_distribution()
    
    with tf.name_scope('sample'):
      self.create_sample_z()
    
    with tf.name_scope('decoder'):
      self.create_decoder()
    
    with tf.name_scope('optimizer'):
      self.create_optimizer()
    
    if restore == False:
      print("Initialization")
      init = tf.global_variables_initializer()
      with tf.Session() as sess:
        self.sess.run(init)

    self.epoch_num = 0
    
    print("Model created!")

  def create_encoder(self):
    print("Creating encoder...")
    activation = self.x
    for i in range(len(self.encoder_arch)-1):
        w = self.get_W([self.encoder_arch[i], self.encoder_arch[i+1]], name="w"+str(i)+"encoder")
        if i == 0:
          self.filter = w
        #tf.summary.histogram("w"+str(i)+"encoder", w)
        b = self.get_b([self.encoder_arch[i+1]], name="b"+str(i)+"encoder")
        activation = self.activation_fun(tf.add(tf.matmul(activation, w), b))
    self.encoding_activation = activation
    

  def create_classifier(self):
    w_classify = self.get_W([self.encoder_arch[-1], 10], name="w-classify")
    b_classify = self.get_b([10], name="b-classify")
    self.pred = tf.nn.softmax(tf.matmul(self.encoding_activation, w_classify)+b_classify)
        
  def create_latent_distribution(self):
    w_mean = self.get_W([self.encoder_arch[-1], self.latent_dim], name="w-latent-mean")
    b_mean = self.get_b([self.latent_dim], name="b-latent-mean")
    self.z_mean = tf.add(tf.matmul(self.encoding_activation, w_mean), b_mean)
    
    w_sigma = self.get_W([self.encoder_arch[-1], self.latent_dim], name="w-latent-sigma")
    b_sigma = self.get_b([self.latent_dim], name="w-latent-sigma")
    self.z_sigma = tf.add(tf.matmul(self.encoding_activation, w_sigma), b_sigma)
    
  def create_sample_z(self):
    eps = tf.random_normal((self.batch_size, self.latent_dim), 0, 1, dtype=tf.float32)
    self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_sigma)), eps))
    
  def create_decoder(self):
    print("Creating decoder...")
    activation = self.z
    for i in range(len(self.decoder_arch)-1):
        w = self.get_W([self.decoder_arch[i], self.decoder_arch[i+1]], name="w"+str(i)+"decoder")
        b = self.get_b([self.decoder_arch[i+1]], name="w"+str(i)+"decoder")
        if i == len(self.decoder_arch)-2:
          activation = tf.nn.sigmoid(tf.add(tf.matmul(activation, w), b))
        else:
          activation = self.activation_fun(tf.add(tf.matmul(activation, w), b))
    self.decoding_reconstruction = activation
    
  def create_optimizer(self):
    print("Creating optimizer...")
    self.reconstr_loss = \
      - tf.reduce_sum(self.x * tf.log(1e-5 + self.decoding_reconstruction) + \
      (1-self.x) * tf.log(1e-5+1 - self.decoding_reconstruction), 1)
    self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_sigma - tf.square(self.z_mean) - tf.exp(self.z_sigma), 1)
    self.loss_rec = tf.reduce_mean(self.reconstr_loss)
    self.loss_kl = tf.reduce_mean(self.latent_loss)
    self.loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss)
    self.optimizer =  tf.train.AdamOptimizer(learning_rate=self.ln_rate).minimize(self.loss)
    tf.summary.scalar('loss', self.loss)

    self.pred_cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.pred,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.pred_optimizer = tf.train.AdamOptimizer(learning_rate=self.ln_rate).minimize(self.pred_cross_entropy)

  def partial_fit(self, X):
    opt, loss, summary, loss_rec, loss_kl = self.sess.run((self.optimizer, self.loss, self.merged, self.loss_rec, self.loss_kl), feed_dict={self.x: X})
    return loss, summary, loss_rec, loss_kl 

  def train(self, learning_rate=0.001, training_epochs=10):
    self.ln_rate = learning_rate
    total_batch = int(n_samples / self.batch_size)
    for epoch in range(training_epochs):
      loss_sum = 0
      for batch in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(self.batch_size)
        loss, summary, loss_rec, loss_kl  = self.partial_fit(batch_xs)
        loss_sum += loss
      self.writer.add_summary(summary, self.epoch_num)
      #if not epoch%10: print("Epoch: {:}, loss: {:.2f}".format(epoch, loss_sum/n_samples)) 
      print("Epoch: {:}, loss: {:.2f}, rec: {:.2f}, kl: {:.3f}".format(self.epoch_num, loss_sum/n_samples, loss_rec, loss_kl)) 
      self.epoch_num += 1

  def train_classifier(self, learning_rate=0.001, training_epochs=10):
    self.ln_rate = learning_rate
    total_batch = int(n_samples / self.batch_size)
    for epoch in range(training_epochs):
      loss_sum = 0
      for batch in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
        loss, acc, acc_test = self.partial_fit_classifier(batch_xs, batch_ys)
        loss_sum += loss
      #if not epoch%10: print("Epoch: {:}, loss: {:.2f}".format(epoch, loss_sum/n_samples)) 
      print("Epoch: {:}, loss: {:.2f}, acc: {:.2f}, test: {:.4f}".format(self.epoch_num, loss_sum/n_samples, acc, acc_test)) 
      self.epoch_num += 1

  def partial_fit_classifier(self, X, Y):
    opt, loss, acc_train = self.sess.run((self.pred_optimizer, self.pred_cross_entropy, self.accuracy), feed_dict={self.x: X, self.y: Y})
    test_x, test_y = mnist.test.next_batch(mnist.test.num_examples)
    test_acc = self.sess.run((self.accuracy), feed_dict={self.x: test_x, self.y: test_y})
    return loss, acc_train, test_acc

  def create_tensorboard(self):
    self.merged = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter(self.log_name)
    
    
  def reconstruct(self, X):
    rec = self.sess.run(self.decoding_reconstruction, feed_dict={self.x: X})
    return rec

  def generate_from_z(self, z):
    rec = self.sess.run(self.decoding_reconstruction, feed_dict={self.z: z})
    return rec

  def get_z(self, X):
    z = self.sess.run(self.z, feed_dict={self.x: X})
    return z

  def generate(self, std=1.0):
    z = np.random.normal(scale=std, size = (self.batch_size, self.latent_dim))
    rec = self.sess.run(self.decoding_reconstruction, feed_dict={self.z: z})
    plt.figure(figsize=(10,10))
    showGrid(rec[0:400])
    plt.savefig('./samples.png', bbox_inches='tight')
    return rec

  def save(self, name='model.ckpt'):
    # save graph
    saver = tf.train.Saver()
    saver.save(self.sess, name)
    # save class instance
    with open(name+'.pkl', 'wb') as pFile:
      pickle.dump((self.log_name, self.epoch_num), pFile, pickle.HIGHEST_PROTOCOL)
  
  def load(self, name='/users/grad/xjiang/vae/model.ckpt'):
    print("Model restored from " + name)
    saver = tf.train.Saver()
    saver.restore(self.sess, name)
    with open(name+'.pkl', 'r') as pFile:
      (self.log_name, self.epoch_num) = pickle.load(pFile)

  def show_latent_space(self):
    z_x_values = np.linspace(3, -3, 20)
    z_y_values = np.linspace(-3, 3, 20)
    z_x, z_y = np.meshgrid(z_x_values, z_y_values)
    z_x = z_x.flatten()
    z_y = z_y.flatten()
    z_values = np.transpose([z_y, z_x])
    images = self.generate_from_z(z_values)
    plt.figure(figsize=(10,10))
    showGrid(images[0:400])
    plt.savefig('./latent_space/{:}.png'.format(self.epoch_num), bbox_inches='tight')
    
    batch_n = 10
    z = np.empty(shape=[self.batch_size*batch_n, 2])
    for i in range(batch_n):
      sample = mnist.train.images[self.batch_size*i:self.batch_size * (i+1)]
      new_z = self.get_z(sample)
      z[self.batch_size*i:self.batch_size * (i+1)] = new_z
    plt.figure(figsize=(10,10))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(mnist.train.labels[0:self.batch_size*batch_n], 1))
    plt.colorbar()
#    plt.grid()
    plt.savefig('./latent_space.png'.format(self.epoch_num), bbox_inches='tight')

  def show_filter(self):
    #self.writer.add_summary(self.filter_vis, self.epoch_num)
    ws = self.filter.eval()
    #ws = np.reshape(ws, ( 28, 28, self.encoder_arch[1]))
    ws = np.transpose(ws, (1, 0))
    showGrid(ws[0:400])
    plt.savefig('./filters.png'.format(self.epoch_num), bbox_inches='tight')

      
def showGrid(images):
  dim = int(math.sqrt(images.shape[0]))
  images = np.reshape(images, (dim*dim, 28, 28))
  canvas = np.zeros([28*dim, 28*dim])
  for i in range(dim):
    for j in range(dim):
      canvas[i*28:(i+1)*28, j*28:(j+1)*28] = images[i*dim+j]
  plt.imshow(canvas, vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))

def visualize():
  vae = VAE(encoder_arch=[28*28, 256], decoder_arch=[256, 28*28], latent_dim=50, ln_rate = 0.001, batch_size=400, sess=sess, restore=True)
  vae.load()
  vae.generate()
#  vae.show_latent_space()
  vae.show_filter()
  

with tf.Session() as sess:
  vis = False
  if vis:
    visualize()
    exit()
  restore = False
  vae = VAE(encoder_arch=[28*28, 500, 200], decoder_arch=[200, 500, 28*28], latent_dim=50, ln_rate = 0.001, batch_size=100, sess=sess, restore=restore)
  if restore:
    vae.load()
  vae.create_tensorboard()
  vae.train(training_epochs=400)

  vae.train_classifier()
  vae.save()
