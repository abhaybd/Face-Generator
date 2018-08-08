import numpy as np

def rescale(data, min_num, max_num, data_min=None, data_max=None):
      if data_min is None:
            data_min = np.min(data)
      if data_max is None:
            data_max = np.max(data)
      data_range = data_max - data_min
      data = ((data - data_min) / data_range) * (max_num - min_num) + min_num
      return data

def write_log(callback, names, logs, epoch):
      import tensorflow as tf
      """Write stats to TensorBoard"""
      for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, epoch)
        callback.writer.flush()