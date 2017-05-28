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

def get_imgid(file):
    return int(file.split('_')[-1].split('.')[0])

def find_files(directory, pattern='*.tif', sortby="auto"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if sortby == "auto":
        files = np.sort(files)
    else:
        files = sorted(files, key=lambda fname: get_imgid(fname))
    return files



def load_image(directory, sortby="img_id"):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    files = find_files(directory, sortby=sortby)
    for filename in files:
        img = skimage.io.imread(filename, plugin='tifffile').astype(np.float32)
        #img = (img - 4000.)/5000. 
        img *= (1./10000)
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
def get_multi(labels):
    lab = np.zeros(14)
    lab[0] = get_weather(labels)
    if "primary" in labels:
        lab[1] = 1
    if "agriculture" in labels:
        lab[2] = 1
    if "water" in labels:
        lab[3] = 1
    if "road" in labels:
        lab[4] = 1
    if "cultivation" in labels:
        lab[5] = 1
    if "habitation" in labels:
        lab[6] = 1
    if "bare_ground" in labels:
        lab[7] = 1
    if "slash_burn" in labels:
        lab[8] = 1
    if "conventional_mine" in labels:
        lab[9] = 1
    if "artisinal_mine" in labels:
        lab[10] = 1
    if "selective_logging" in labels:
        lab[11] = 1
    if "blooming" in labels:
        lab[12] = 1
    if "blow_down" in labels:
        lab[13] = 1
    return lab

def load_label_df(filename, multi=True):
    df_train = pandas.DataFrame.from_csv(filename)
    df_train['slabels'] = df_train['tags'].apply(lambda x: x.split(' '))
    if multi:
        df_train['labels'] = df_train['slabels'].apply(lambda row: get_multi(row))
    else:
        df_train['labels'] = df_train['labels'].apply(lambda row: get_weather(row))
    return df_train


class Reader(object):
    '''Generic background reader that preprocesses files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 coord,
                 train=True, 
                 threshold=None,
                 queue_size=16, 
                 min_after_dequeue=4,
                 q_shape=None,
                 pattern='*.tif', 
                 n_threads=2,
                 multi=True,
                 label_file="./Data/train/train_v2.csv"):
        self.data_dir = data_dir
        self.coord = coord
        self.n_threads = n_threads
        self.threshold = threshold
        self.ftype = pattern
        self.corpus_size = get_corpus_size(self.data_dir, pattern=self.ftype)
        self.threads = []
        self.multi = multi
        self.train = train
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        if multi and train:
            self.label_shape =[14]
        else:
            self.label_shape = []
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=self.label_shape, name='label') #!!!
        
        
        if self.train:
            self.labels_df = load_label_df(label_file, multi=multi)
            if q_shape:
                #self.queue = tf.FIFOQueue(queue_size,[tf.float32,tf.int32], shapes=[q_shape,[]])
                self.queue = tf.RandomShuffleQueue(queue_size, min_after_dequeue,
                    [tf.float32,tf.int32], shapes=[q_shape, self.label_shape])
            else:
                self.q_shape = [(1, None, None, 4)]
                self.queue = tf.PaddingFIFOQueue(queue_size,
                                                 [tf.float32, tf.int32],
                                                 shapes=[self.q_shape,self.label_shape])
        else:
            self.labels_df = None
            self.queue = tf.FIFOQueue(queue_size,[tf.float32,tf.int32], shapes=[q_shape,[]])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.label_placeholder])

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
                if self.train:
                    try: 
                        label = self.labels_df['labels'][img_id]
                    except(KeyError):
                        print('No match for ', img_id)
                        continue
                else:
                    label = np.int32(img_id.split('_')[1])
                if self.coord.should_stop():
                    stop = True
                    break
                if self.threshold is not None:
                    #TODO:  Perform quality check if needed
                    pass

                sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: img, self.label_placeholder: label})
                    
    def start_threads(self, sess):
        for _ in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
