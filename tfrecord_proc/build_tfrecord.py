import tensorflow as tf
import numpy as np
import math
import argparse
import os
import re

parser = argparse.ArgumentParser(description='create tfrecord')
parser.add_argument('--datadir',default='/home/sensetime/dataset/cornell-grasp-data/raw-data',type=str,help='dataset path')
parser.add_argument('--outpath',default='/home/sensetime/mdn/dataset',type=str,help='output path')
#parser.add_argument('',default='',type=,help='')
args=parser.parse_args()

subsets = ['Train','Test']
categories = ['positive','negative']
POSpattern = 'pcd{}cpos.txt'
NEGpattern = 'pcd{}cneg.txt'

def _int64_feature(value):
    """

    :param value: an int type data, if string please refer to tf.train.Int64List.fromstring()
    :return: int64list feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    """

    :param value: value is orignal a list
    :return: int64list feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _byte_feature(value):
    """
    generatre byte list feature
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _byte_feature_list(value):
    """

    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """

    :param value:
    :return:
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _float_feature_list(value):
    """

    :param value:
    :return:
    """
    return  tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_box(vertex_list):
    """

    :param vertex_list: four vertex
    :return: a np array indentify the bounding box: center, angle, width, height
    """
    x = [np.float(vertex_list[i][0]) for i in range(len(vertex_list))]
    y = [np.float(vertex_list[i][1]) for i in range(len(vertex_list))]

    #print vertex_list
    center_x = np.mean(x)
    center_y = np.mean(y)
    min_x = [i for i,x_i in enumerate(x) if x_i == min(x)]
    max_y = [i for i,y_i in enumerate(y) if y_i == max(y)]
    min_y = [i for i,y_i in enumerate(y) if y_i == min(y)]
    if not (len(min_x) == 1):
        min_x = [min_x[0]] if y[min_x[0]] > y[min_x[1]] else [min_x[1]]
    if not (len(max_y) == 1):
        max_y = [max_y[0]] if x[max_y[0]] > x[max_y[1]] else [max_y[1]]
        min_y = [min_y[0]] if x[min_y[0]] > y[min_y[1]] else [min_y[1]]
    ang = y[max_y[0]]-y[min_x[0]] / x[max_y[0]] - x[min_x[0]]
    width = math.sqrt(math.pow(y[max_y[0]]-y[min_x[0]],2)+math.pow(x[max_y[0]] - x[min_x[0]],2))
    height =  math.sqrt(math.pow(y[min_y[0]]-y[min_x[0]],2)+math.pow(x[min_y[0]] - x[min_x[0]],2))
    ang = np.tanh(ang)
    center_x = center_x / 100.0
    center_y = center_y / 100.0
    box = np.array([center_x,center_y,ang,width,height])
    box.astype(np.float32)
    return box

def read_box(sampath,boxfile):
    """

    :param sampath:
    :param boxfile:
    :return:
    """
    with tf.gfile.GFile(os.path.join(sampath,boxfile)) as fid:
        lines = fid.readlines()
    lines = [line.strip().split(' ') for line in lines]
    #print lines
    sampnum = len(lines)/4
    box_list = [convert_box([lines[i*4],lines[i*4+1],lines[i*4+2],lines[i*4+3]]) for i in range(sampnum)]
    return box_list


def create_sample(img,posbox_list,negbox_list,img_id):
    """

    :param img: original image
    :param label: the label indicate positive or negative boundg box
    :param posbox_list: a 5 dimension vector identify the bounding box
    :param negbox_list: a 5 dimension vector identify the bounding box
    :param img_id: image id
    :return: a protobuf defined by tf.train.Example() or tf.train.SequenceExample()
    """
    example = tf.train.SequenceExample(
        context = tf.train.Features(
            feature={
                'img':   _byte_feature(img),
                'img_id':_byte_feature_list(img_id)
            }),
        feature_lists = tf.train.FeatureLists(
            feature_list={
                'posbox': tf.train.FeatureList(feature = [_float_feature_list(posbox_list[i]) for i in range(len(posbox_list))]),
                'negbox': tf.train.FeatureList(feature = [_float_feature_list(negbox_list[i]) for i in range(len(negbox_list))])
            })
    )
    return example

def main():
    if not os.path.exists(args.datadir):
        raise ValueError("data dir {} not exist".format(args.datadir))


    for set in subsets:
        imgpath = os.path.join(args.datadir,set,'Images')
        img_set = tf.gfile.Glob(os.path.join(imgpath,'*.png'))
        tfrecord_path = os.path.join(args.outpath,'{}.tfrecord'.format(set))
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for img in img_set:
            img_name = os.path.basename(img)
            img_id = re.findall('\d+',img_name)
            sampath = os.path.join(args.datadir,set)
            posboxfile = POSpattern.format(img_id[0])
            negboxfile = NEGpattern.format(img_id[0])
            posbox_list = read_box(os.path.join(sampath,'positive'),posboxfile)
            negbox_list = read_box(os.path.join(sampath,'negative'),negboxfile)
            encoded_img = tf.gfile.GFile(os.path.join(imgpath,img)).read()
            sample = create_sample(encoded_img,posbox_list,negbox_list,img_id)
            writer.write(sample.SerializeToString())
        writer.close()
        print set

if __name__=='__main__':
    main()
