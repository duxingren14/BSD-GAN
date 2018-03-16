import os
import scipy.misc
import numpy as np

from model import BranchGAN
from utils import pp, visualize, to_json, show_all_variables, imread

import tensorflow as tf
from glob import glob


flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 20, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 256, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", None, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celeba_hq256", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("z_dim", 30, "Dimensions of z [50]")
flags.DEFINE_boolean("use_z_pyramid", True, "True for using z pyramid")
flags.DEFINE_boolean("use_residual_block", False, "True for using residual block")
flags.DEFINE_boolean("use_two_stage_training", True, "True for using two-stage training at each epoch")
flags.DEFINE_boolean("random_flip", False, "True for randomly flipping training images")
flags.DEFINE_boolean("random_crop", False, "True for randomly cropping training images")
flags.DEFINE_boolean("random_rotate", False, "True for randomly rotating training images")


FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not FLAGS.input_width:
    exit("[Exit] input_width is None. please use flag '--input_width' to specify the input image width.")
  else:
    if not FLAGS.output_width:
      FLAGS.output_width = FLAGS.input_width
      FLAGS.crop = False
    elif FLAGS.output_width < FLAGS.input_width:
      FLAGS.crop = True
    elif FLAGS.output_width == FLAGS.input_width:
      FLAGS.crop = False
    elif FLAGS.output_width > FLAGS.input_width:
      exit("[Exit] output_width should be smaller than or equal to input_width")

  if FLAGS.input_height is None:
    FLAGS.input_height = FLAGS.input_width

  if FLAGS.output_height is None:
    FLAGS.output_height = FLAGS.input_height
  elif FLAGS.output_height > FLAGS.input_height:
    exit("[Exit] output_height should be smaller than or equal to input_height")
  elif FLAGS.output_height < FLAGS.input_height:
    FLAGS.crop = True

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    branchgan = BranchGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir, 
          use_z_pyramid=FLAGS.use_z_pyramid, 
          use_residual_block = FLAGS.use_residual_block,
          use_two_stage_training=FLAGS.use_two_stage_training,
          random_crop=FLAGS.random_crop,
          random_flip=FLAGS.random_flip,
          random_rotate=FLAGS.random_rotate,
          epoch = FLAGS.epoch)

    show_all_variables()

    if FLAGS.train:
      branchgan.train(FLAGS)
    else:
      if not branchgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      visualize(sess, branchgan, FLAGS, option=1, name=FLAGS.dataset)


if __name__ == '__main__':
  tf.app.run()
