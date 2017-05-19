import fnmatch
import os
import re
import threading
# import cv2
import pandas
import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform


#G
def get_corpus_size(directory, pattern='*.tif'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return len(files)

def find_files(directory, pattern='*.tif'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return np.sort(files)


def load_image(directory):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    files = find_files(directory)
    for filename in files:
        img = skimage.io.imread(filename, plugin='tifffile').astype(np.float32)
        #img = (img - 4000.)/5000. 
        img *= (256./10000)
        img -= 128
        img_id = filename.split('/')[-1].split('.')[0]
        yield img, img_id


def get_weather(labels):
    if "cloudy" in labels:
        return 0
    elif "partly_cloudy" in labels:
        return 1
    elif "haze" in labels:
        return 2
    elif "clear" in labels:
        return 3
    else:
        print(labels)
        #raise Exception("weather info not available")
        return 3

def load_label_df(filename):
    df_train = pandas.DataFrame.from_csv(filename)
    df_train['labels'] = df_train['tags'].apply(lambda x: x.split(' '))
    df_train['weather'] = df_train['labels'].apply(lambda row: get_weather(row))
    return df_train


class Reader(object):
    '''Generic background reader that preprocesses files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 coord,
                 threshold=None,
                 queue_size=16, 
                 q_shape=None,
                 pattern='*.tif', 
                 n_threads=1):
        self.data_dir = data_dir
        self.coord = coord
        self.n_threads = n_threads
        self.threshold = threshold
        self.ftype = pattern
        self.corpus_size = get_corpus_size(self.data_dir, pattern=self.ftype)
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name='label') #!!!
        if q_shape:
            self.queue = tf.FIFOQueue(queue_size,[tf.float32,tf.int32], shapes=[q_shape,[]])
        else:
            self.q_shape = [(1, None, None, 4)]
            self.queue = tf.PaddingFIFOQueue(queue_size,
                                             [tf.float32, tf.int32],
                                             shapes=[self.q_shape,[]])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.label_placeholder])
        self.labels_df = load_label_df("./Data/train/train_v2.csv")

    def dequeue(self, num_elements):
        images, labels = self.queue.dequeue_many(num_elements)
        return images, labels

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_image(self.data_dir)
            for img, img_id in iterator:
                #print(filename)
                try: 
                    label = self.labels_df['weather'][img_id]
                except(KeyError):
                    print('No match for ', img_id)
                    continue
                if self.coord.should_stop():
                    stop = True
                    break
                if self.threshold is not None:
                    #TODO:  Perform quality check if needed
                    pass

                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: img, self.label_placeholder: label})
                    
    def start_threads(self, sess):
        for _ in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
