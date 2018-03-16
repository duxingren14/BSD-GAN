from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.python.ops import data_flow_ops


from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class BranchGAN(object):
  def __init__(self, sess, input_height=256, input_width=256, crop=False,
         batch_size=20,  output_height=256, output_width=256,
         z_dim=30, use_z_pyramid=False, 
         gf_dim=64, df_dim=64, use_residual_block = False, 
         c_dim=3, dataset_name='celeb_hq256',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, 
         use_two_stage_training=False, epoch = 20, 
         random_crop=False, random_flip=False, random_rotate=False):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = int(math.sqrt(self.batch_size))**2
    self.random_rotate = random_rotate

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.use_z_pyramid = use_z_pyramid
    self.use_residual_block = use_residual_block
    self.use_two_stage_training = use_two_stage_training

    self.random_flip= random_flip
    self.random_crop= random_crop
    self.epoch = epoch

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir


    self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
    print('total #imgs: ',len(self.data))
    self.train_size = len(self.data)
    imreadImg = imread(self.data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
    else:
        self.c_dim = 1

    height_pyramid = [self.output_height]
    width_pyramid = [self.output_width]
    while height_pyramid[-1]>8 and width_pyramid[-1]>8:
      height_pyramid.append(conv_out_size_same(height_pyramid[-1], 2))
      width_pyramid.append(conv_out_size_same(width_pyramid[-1], 2))
    self.height_pyramid, self.width_pyramid = list(reversed(height_pyramid)), list(reversed(width_pyramid))
    print('height_pyramid: ', height_pyramid)
    print('width_pyramid: ', width_pyramid)
    self.n_levels = len(height_pyramid)
    self.build_model()

  
  def build_model(self):
    self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
    input_queue = data_flow_ops.FIFOQueue(capacity=200000,
                                    dtypes=[tf.string],
                                    shapes=[(1,)],
                                    shared_name=None, name=None)
    self.enqueue_op = input_queue.enqueue_many([self.image_paths_placeholder], name='enqueue_op')
    nrof_preprocess_threads = 4
    images_all = []
    for _ in range(nrof_preprocess_threads):
        filenames = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=self.c_dim)
            #image = tf.image.resize_images(image, [self.input_height, self.input_width])
            if self.random_rotate:
                image = tf.py_func(random_rotate_image, [image], tf.uint8)
            if self.random_crop:
                image = tf.random_crop(image, [self.output_height, self.output_width, self.c_dim])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, self.output_height, self.output_width)
            if self.random_flip:
                image = tf.image.random_flip_left_right(image)
            
            image.set_shape((self.output_height, self.output_width, self.c_dim))
            images.append(tf.cast(image, tf.float32)/127.5-1.0)#tf.image.per_image_standardization(image))
        images_all.append(images)
    
    image_batch = tf.train.batch_join(
            images_all, batch_size=self.batch_size, enqueue_many=False,
            capacity=4 * nrof_preprocess_threads * self.batch_size,
            allow_smaller_final_batch=False)
    self.image_batch = tf.identity(image_batch, 'image_batch')
    
    self.dims = []
    if self.use_z_pyramid:
      self.z_pyramid = []
      for level in range(self.n_levels-1):
        self.dims.append(self.z_dim)
        self.z_pyramid.append(tf.placeholder(tf.float32, [self.batch_size, self.z_dim],  name='z'+str(level)))
      self.z = self.z_pyramid
    else:
      self.z = tf.placeholder(
          tf.float32, [self.batch_size, self.z_dim], name='z')

    if self.use_z_pyramid:
      self.z_sum = histogram_summary("z[level=0]", self.z_pyramid[0])
    else:
      self.z_sum = histogram_summary("z", self.z)

    if self.use_z_pyramid:
      self.G_pyramid = self.generator(self.z_pyramid)
    else:
      self.G_pyramid = self.generator(self.z)

    self.D_real_pyramid = self.discriminator(self.image_batch)
    self.D_fake_pyramid = self.discriminator(self.G_pyramid)

    t_vars = tf.trainable_variables()

    self.d_vars_pyramid = self.get_d_vars_pyramid(t_vars)
    self.g_vars_lastlayer_pyramid, self.g_vars_finetune_pyramid= self.get_g_vars_pyramid(t_vars)

    self.build_losses()

    self.saver = tf.train.Saver()




  def build_optimizers(self, config):
    self.d_optimizer_pyramid = [None]
    self.g_lastlayers_optimizer_pyramid = [None]
    self.g_finetune_optimizer_pyramid = [None]
    for start_level in range(self.n_levels):
      if start_level == 0 :
        continue
      d_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss_pyramid[start_level], var_list=self.d_vars_pyramid[start_level])

      g_lastlayers_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss_pyramid[start_level], \
                var_list= self.g_vars_lastlayer_pyramid[start_level])

      g_finetune_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss_pyramid[start_level], \
                var_list=self.g_vars_finetune_pyramid[start_level] )

      self.d_optimizer_pyramid.append(d_optimizer)

      self.g_lastlayers_optimizer_pyramid.append(g_lastlayers_optimizer)
      self.g_finetune_optimizer_pyramid.append(g_finetune_optimizer)



  def get_d_vars_pyramid(self, t_vars):
    d_vars_pyramid = []
    for start_level in range(self.n_levels):
      if start_level == 0:
        d_vars_pyramid.append(None)
        continue
      d_vars = []
      for var in t_vars:
        if "d_in_"+str(start_level-1) in var.name:
          d_vars.append(var)
        elif "d_lin_"+str(start_level-1) in var.name:
          d_vars.append(var)
        elif  "d_h_"+str(start_level-1)+'_' in var.name:
          d_vars.append(var)
        elif  "d_bn_"+str(start_level-1) in var.name:
          d_vars.append(var)
        elif "d_h_"+str(start_level-2)+'/' in var.name:
          d_vars.append(var)
      if d_vars_pyramid[-1]:
        d_vars_pyramid.append(d_vars + \
            [var for var in d_vars_pyramid[-1] if 'd_in_' not in var.name])
      else:
          d_vars_pyramid.append(d_vars)
    return d_vars_pyramid


  def get_g_vars_pyramid(self, t_vars):
    g_vars_lastlayer_pyramid = []
    g_vars_finetune_pyramid = []
    for start_level in range(self.n_levels):
      if start_level == 0:
        g_vars_lastlayer_pyramid.append(None)
        g_vars_finetune_pyramid.append(None)
        continue
      g_vars = []
      for var in t_vars:
        if "g_zh_"+str(start_level-1) in var.name:
          g_vars.append(var)
        elif "g_h_"+str(start_level-1)+'_' in var.name:
          g_vars.append(var)
        elif "g_bn_"+str(start_level-1) in var.name:
          g_vars.append(var)
        elif "g_h_"+str(start_level-1)+'/' in var.name:
          g_vars.append(var)
        elif  "g_o_"+str(start_level) in var.name:
          g_vars.append(var)
      g_vars_lastlayer_pyramid.append(g_vars)
      if g_vars_finetune_pyramid[-1]:
        g_vars_finetune_pyramid.append(g_vars + \
            [var for var in g_vars_finetune_pyramid[-1] if 'g_o_' not in var.name ])
      else:
          g_vars_finetune_pyramid.append(g_vars)
    return g_vars_lastlayer_pyramid, g_vars_finetune_pyramid

  def build_losses(self):
    self.d_loss_pyramid = [None]
    self.g_loss_pyramid = [None]
    self.d_summary_pyramid = [None]
    self.g_summary_pyramid = [None]

    for start_level in range(self.n_levels):
      if start_level ==0:
          continue
      d_sum, g_sum = [], []
      d_real_loss = celoss(self.D_real_pyramid[start_level], 1)
      d_fake_loss = celoss(self.D_fake_pyramid[start_level], 0)
      g_loss = celoss(self.D_fake_pyramid[start_level], 1)
      d_sum.append(scalar_summary(
            str(start_level)+'/D_real_mean', 
            tf.reduce_mean(tf.sigmoid(self.D_real_pyramid[start_level])))
        )
      g_sum.append(scalar_summary(
            str(start_level)+'/D_fake_mean', 
            tf.reduce_mean(tf.sigmoid(self.D_fake_pyramid[start_level])))
        )
      d_sum.append(scalar_summary(str(start_level)+'/d_real_loss', d_real_loss))
      g_sum.append(scalar_summary(str(start_level)+'/d_fake_loss', d_fake_loss))
      d_sum.append(scalar_summary(str(start_level)+'/d_loss', d_real_loss + d_fake_loss))
      g_sum.append(scalar_summary(str(start_level)+'/g_loss', g_loss))
      g_sum.append(image_summary(str(start_level)+"/G", self.G_pyramid[start_level]))
      d_sum.append(self.z_sum)
      g_sum.append(self.z_sum)

      self.d_summary_pyramid.append(merge_summary(d_sum))
      self.g_summary_pyramid.append(merge_summary(g_sum))
      self.d_loss_pyramid.append(d_real_loss + d_fake_loss)
      self.g_loss_pyramid.append(g_loss)

  def print_var_lists(self):
    for level in range(1,self.n_levels):
      for var in self.g_vars_lastlayer_pyramid[level]:
        print('g lastlayer vars of level ', level, var.name)
      for var in self.d_vars_pyramid[level]:
        print('d vars of level ', level, var.name)
      for var in self.g_vars_finetune_pyramid[level]:
        print('g finetune vars of level ', level, var.name)



  def train(self, config):

    self.build_optimizers(config)
    self.print_var_lists()

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.writer = SummaryWriter("./logs", self.sess.graph)

    if self.use_z_pyramid:
      sample_z = []
      sample_z.append(np.random.uniform(-1, 1, [config.batch_size, self.dims[0]]).astype(np.float32))
      sample_z_dict = {self.z_pyramid[0]:sample_z[0]}
      for level_tmp in range(1, self.n_levels-1, 1):
        sample_z.append(np.random.uniform(-1, 1, [config.batch_size, self.dims[level_tmp]]).astype(np.float32))
        sample_z_dict [self.z_pyramid[level_tmp]] = sample_z[level_tmp]
    else:
      sample_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
      sample_z_dict = {self.z: sample_z}

    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      counter = 1
      print(" [!] Load failed...")

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=self.sess)

    for level in range(1, self.n_levels, 1):#self.n_levels):  counter//config.epoch+
      switch_point = config.epoch // 3
      for epoch in xrange(config.epoch):
        self.data = glob(os.path.join(
          "./data", config.dataset, self.input_fname_pattern))
        np.random.shuffle(self.data)
        self.data = np.array(self.data)
        #print(self.data.shape)
        self.data = np.expand_dims(self.data, axis = 1)
        self.sess.run(self.enqueue_op, {self.image_paths_placeholder: self.data})

        batch_idxs = min(len(self.data), self.train_size) // self.batch_size

        d_optimizer_tmp = self.d_optimizer_pyramid[level]
        if self.use_two_stage_training:
          if epoch < config.epoch //3:
            g_optimizer_tmp = self.g_lastlayers_optimizer_pyramid[level]
          else:
            g_optimizer_tmp = self.g_finetune_optimizer_pyramid[level]
        else:
          g_optimizer_tmp = self.g_finetune_optimizer_pyramid[level]
        
        for idx in xrange(0, batch_idxs):
            print('building z dictionary...............')
            if self.use_z_pyramid:
              z_dict = {}
              for level_tmp in range(self.n_levels-1):
              	if level_tmp < level:
                  batch_z = np.random.uniform(-1, 1, [config.batch_size, self.dims[level_tmp]]).astype(np.float32)
                else:
                  batch_z = np.zeros([config.batch_size, self.dims[level_tmp]], dtype = np.float32)
                if level_tmp == level -1  and level_tmp > 0:
                  if self.use_two_stage_training:
                    if epoch < switch_point:
                      batch_z = np.zeros([config.batch_size, self.dims[level_tmp]], dtype = np.float32)
                    else:
                      factor = (epoch - switch_point) / (config.epoch - switch_point)
                      batch_z = np.random.uniform(-factor, factor, [config.batch_size, self.dims[level_tmp]]).astype(np.float32)
                z_dict [self.z_pyramid[level_tmp]] = batch_z
            else:
              batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
              z_dict = {self.z: batch_z}
            train_dict = z_dict

            # Update D network
            _, errD, errG, summary_str = self.sess.run([d_optimizer_tmp, \
                 self.d_loss_pyramid[level], \
                 self.g_loss_pyramid[level], \
                 self.d_summary_pyramid[level]],\
              feed_dict=train_dict)
            self.writer.add_summary(summary_str, counter)

            # Update G network
            _, summary_str = self.sess.run([g_optimizer_tmp, self.g_summary_pyramid[level]],
              feed_dict=z_dict)
            self.writer.add_summary(summary_str, counter)

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            _, summary_str = self.sess.run([g_optimizer_tmp, \
                self.g_summary_pyramid[level]],
              feed_dict=z_dict)
            self.writer.add_summary(summary_str, counter)

            counter += 1
            print("level=%2d epoch=%2d batch_nd=%4d/%4d time=%4.4f, d_loss=%.8f, g_loss=%.8f" \
              % (level, epoch, idx, batch_idxs,
              time.time() - start_time, errD, errG))

            if np.mod(counter, 500) == 1:
              try:
                samples, g_loss = self.sess.run(
                  [self.G_pyramid[level], self.g_loss_pyramid[level]],
                  feed_dict=sample_z_dict,
                )
                save_images(samples[0:self.sample_num], image_manifold_size(samples[0:self.sample_num].shape[0]),
                    './{}/train_{:02d}_{:02d}_{:04d}.png'.format(config.sample_dir, level, epoch, idx))
                print("[Sample]  g_loss: %.8f" % (g_loss)) 
              except:
                print("one pic error!...")
            if np.mod(counter, 500) == 2:
              self.save(config.checkpoint_dir, counter)


  def discriminator(self, inputs):
      D_pyramid = [None]
      if not isinstance(inputs, list):
          for start_level in range(1, self.n_levels, 1):
              height, width = self.height_pyramid[start_level], self.width_pyramid[start_level]
              D = self.sub_discriminator(tf.image.resize_images(inputs, [height, width]), start_level)
              D_pyramid.append(D)
      else:
	      for start_level in range(1, self.n_levels, 1):
	          D = self.sub_discriminator(inputs[start_level], start_level)
	          D_pyramid.append(D)
      return list(D_pyramid)


  def sub_discriminator(self, input_tensor, start_level):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
      h_pyramid = []
      for level in range(start_level-1, -1, -1):
          height, width = self.height_pyramid[level], self.width_pyramid[level]
          df_dim = self.get_f_dim(self.df_dim, height, width)

          if h_pyramid:
              h = conv2d(h_pyramid[-1], df_dim, name='d_h_'+str(level))
          else:
              h = conv2d(input_tensor, df_dim, name='d_in_'+str(level))
          h = self.residual_block(h, level, name='d_')
          bn = instance_batch_norm(name='d_bn_'+str(level))
          h = lrelu(bn(h))
          if level == 0:
              D = linear(tf.reshape(h, [self.batch_size, -1]), 1, 'd_lin_'+str(level))
          h_pyramid.append(h)
      return D

  def residual_block(self, inputs, level, name='d_'):
    if self.use_residual_block:
      df_dim = inputs.get_shape()[3]
      bn1 = instance_batch_norm(name=name+'bn_'+str(level)+'_1')
      h1 = lrelu(bn1(inputs)) #tf.nn.
      h2 = conv2d(h1, df_dim, d_h=1, d_w=1, name=name + 'h_'+str(level)+"_1")

      bn2 = instance_batch_norm(name=name+'bn_'+str(level)+'_2')
      h3 = conv2d(lrelu(bn2(h2)), df_dim, d_h=1, d_w=1, name=name + 'h_'+str(level)+"_2")
      return h3+inputs
    else:
      return inputs

  def generator(self, z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
      h_pyramid,G_pyramid = [],[]
      if isinstance(z, list):
          batch_size = z[0].get_shape().as_list()[0]
      else:
          batch_size = z.get_shape().as_list()[0]

      if isinstance(z, list):
        for level in range(self.n_levels):
          height, width= self.height_pyramid[level], self.width_pyramid[level]
          gf_dim = self.get_f_dim(self.gf_dim, height, width)
          if level == 0:
            z_temp = []
            for lv in range(len(z)):
              z_temp.append(z[lv]) #/(4**lv)
            z_concat = tf.concat(z_temp, axis=1)
            hz, w, b = linear(z_concat, gf_dim * height * width, 'g_zh_'+str(level), with_w=True)
            hz = tf.reshape(hz, [batch_size, height, width, gf_dim])
            bn = instance_batch_norm(name='g_bn_'+str(level))
            h = lrelu(bn(hz)) 
            G = None
          elif level == self.n_levels - 1:
            G, w, b = deconv2d(h_pyramid[-1], [batch_size, height, width, self.c_dim], name='g_o_'+str(level), with_w=True)
            G = tf.nn.tanh(G)
            h, hz = None, None
          else:
            h, w, b = deconv2d(h_pyramid[-1], [batch_size, height, width, gf_dim], name='g_h_'+str(level), with_w=True)
            
            #h = tf.concat([h , hz], axis=3)
            h = self.residual_block(h,  level, name='g_')
            bn = instance_batch_norm(name='g_bn_'+str(level))
            h = lrelu(bn(h))  #tf.nn.
            G, w, b = deconv2d(h_pyramid[-1], [batch_size, height, width, self.c_dim], name='g_o_'+str(level), with_w=True)
            G = tf.nn.tanh(G)
          h_pyramid.append(h)
          G_pyramid.append(G)
      else:
        for level in range(self.n_levels):
          height, width = self.height_pyramid[level], self.width_pyramid[level]
          gf_dim = self.get_f_dim(self.gf_dim, height, width)
          if level == 0:
            hz, w, b = linear(z, gf_dim * height * width, 'g_zh_'+str(level), with_w=True)
            hz = tf.reshape(hz, [batch_size, height, width, gf_dim])
            bn = instance_batch_norm(name='g_bn_'+str(level))
            h = lrelu(bn(hz)) 

            G = None
          elif level == self.n_levels - 1:
            G, w, b = deconv2d(h_pyramid[-1], [batch_size, height, width, self.c_dim], name='g_o_'+str(level), with_w=True)
            G = tf.nn.tanh(G)
            h, hz = None, None
          else:
            hz=None
            h, w, b = deconv2d(h_pyramid[-1], [batch_size, height, width, gf_dim], name='g_h_'+str(level), with_w=True)
            
            h = self.residual_block(h, level, name='g_')
            bn = instance_batch_norm(name='g_bn_'+str(level))
            h = lrelu(bn(h)) #tf.nn.

            G, w, b = deconv2d(h_pyramid[-1], [batch_size, height, width, self.c_dim], name='g_o_'+str(level), with_w=True)
            G = tf.nn.tanh(G)
          h_pyramid.append(h)
          G_pyramid.append(G)
      return G_pyramid
  
  def get_f_dim(self, f_dim_base, height, width):
      times = int(64//((height + width)//2))
      if times > 8:
          times = 8
      elif times < 1:
          times = 1
      return times * f_dim_base
  
  @property
  def model_dir(self):
    return "{}-{}-{}-{}_{}-{}_{}-{}-{}_{}-{}-{}".format(
        self.dataset_name, self.n_levels, 
        self.output_height, self.output_width,
        self.use_z_pyramid, self.z_dim,
        self.df_dim, self.gf_dim, self.use_residual_block,
        self.batch_size, self.epoch, 
        self.use_two_stage_training
        )
      
  def save(self, checkpoint_dir, step):
    model_name = "branchgan.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0