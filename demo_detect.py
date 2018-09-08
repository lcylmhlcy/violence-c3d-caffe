#!/usr/bin/env python
#-*- coding: UTF-8 -*-  

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# plt.rcParams['image.cmap'] = 'gray'

#编写一个函数 ，用于显示各层数据
def show_data(data,jpg_name,padsize=0,padval=0):
    data -= data.min()
    data /= data.max()
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data)
    plt.axis('off')
    plt.savefig('/home/computer/video-caffe/examples/violence_c3d/detect/test'+jpg_name)

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
import os

caffe_root = '/home/computer/video-caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
os.environ['GLOG_minloglevel'] = '3'
caffe_root = os.path.expanduser('~/video-caffe/') # change with your install location
sys.path.insert(0, os.path.join(caffe_root, 'python'))
sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/proto'))
sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/tripletloss'))

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os

model_weights = './examples/violence_c3d/violence_c3d_iter_1000.caffemodel'
model_def = './examples/violence_c3d/c3d_deploy.prototxt'

if not os.path.isfile(model_weights):
    print "[Error] model weights can't be found."
    sys.exit(-1)

print 'model found.'

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)



##################################### create transformer for the input called 'data' ##########################

# # load the mean ImageNet image (as distributed with Caffe) for subtraction
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# convert from HxWxCxL to CxLxHxW (L=temporal length)
length = 40
transformer.set_transpose('data', (2,3,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          40,        # length of a clip
                          60, 90)  # image size



#####################################################input 40 image ###########################################

# clip = np.tile(
#         caffe.io.load_image('./examples/violence_c3d/detect/111/violence/29.jpg'),
#         (40,1,1,1)
#         )
# print clip.shape
# clip = np.transpose(clip, (1,2,3,0))
# print clip.shape
# # print "clip.shape={}".format(clip.shape)

temp_pic_data1 = caffe.io.load_image('./examples/violence_c3d/detect/fi100_xvid/image_0001.jpg')
height = temp_pic_data1.shape[0]
width = temp_pic_data1.shape[1]
channel = temp_pic_data1.shape[2]

clip_1 = [None] * channel
clip_2 = [clip_1] * width
clip_3 = [clip_2] * height
clip = [clip_3] * 40

for i in range(1,10):
    temp_pic = str('./examples/violence_c3d/detect/fi100_xvid/image_000') +str(i) + str('.jpg')
    temp_pic_data = caffe.io.load_image(temp_pic)
    clip[i-1] = temp_pic_data


for i in range(10,41):
    temp_pic = str('./examples/violence_c3d/detect/fi100_xvid/image_00') +str(i) + str('.jpg')
    temp_pic_data = caffe.io.load_image(temp_pic)
    clip[i-1] = temp_pic_data

clip = np.array(clip)
#print clip.shape
clip = np.transpose(clip, (1,2,3,0))
print clip.shape

transformed_image = transformer.preprocess('data', clip)
# plt.imshow(transformed_image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image
# for l in range(0, length):
#     print "net.blobs['data'].data[0,:,{},:,:]={}".format(l,net.blobs['data'].data[0,:,l,:,:])



#################################################### perform classification ##################################
# output = net.forward()
net.forward()

# print '\n'

# print '输出数据' + '\n'
# for k,v in net.blobs.items():
#     print str(k) + ': ' + str(v.data.shape) + '\n'

# print '\n'

# print '参数数据' + '\n'
# for k,v in net.params.items():
#     print str(k) + ': ' + str(v[0].data.shape) + '\n'
# print '\n'

# 显示第一个卷积层的输出数据和权值（filter）
data_temp = net.blobs['conv1a'].data[0]
for i in range(0,7):
    for j in range(0,35):
        file_temp = 'conv1a_' + str(i) + '_' + str(j) + '.jpg'
        plt.imshow(data_temp[i][j][:][:])
        plt.axis('off')
        plt.savefig('/home/computer/video-caffe/examples/violence_c3d/detect/test/' + file_temp)
        plt.close('all')
    # show_data(data_temp[i][:][:][:],file_temp)
print net.blobs['conv1a'].data.shape

# #显示第一次pooling后的输出数据
# data_temp = net.blobs['pool1'].data[0]
# for i in range(0,7):
#     file_temp = 'pool1_' + str(i) + '.jpg'
#     show_data(data_temp[i][:][:][:],file_temp)
# print net.blobs['pool1'].data.shape

# #显示第二个卷积层的输出数据
# data_temp = net.blobs['conv2a'].data[0]
# for i in range(0,35):
#     file_temp = 'conv2a_' + str(i) + '.jpg'
#     show_data(data_temp[i][:][:][:],file_temp)
# print net.blobs['conv2a'].data.shape

# #显示第二次pooling后的输出数据
# data_temp = net.blobs['pool2'].data[0]
# for i in range(0,35):
#     file_temp = 'pool2_' + str(i) + '.jpg'
#     show_data(data_temp[i][:][:][:],file_temp)
# print net.blobs['pool2'].data.shape

# #显示第三个卷积层的输出数据
# data_temp = net.blobs['conv3a'].data[0]
# for i in range(0,5):
#     file_temp = 'conv3a_' + str(i) + '.jpg'
#     show_data(data_temp[i][:][:][:],file_temp)
# print net.blobs['conv3a'].data.shape


#################################################### output class ##############################################

# output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
# print 'predicted class is:', output['prob'][0].argmax()
print 'predicted class is:', net.blobs['prob'].data[0]
print 'predicted class is:', net.blobs['prob'].data[0].argmax()



########################################################################################
# for i in range(1,51):
#     temp_file = str('./examples/violence_c3d/detect/111/no-violence/') + str(i) + str('.jpg')
#     clip = np.tile(caffe.io.load_image(temp_file),(40,1,1,1))
#     clip = np.transpose(clip, (1,2,3,0))
#     transformed_image = transformer.preprocess('data', clip)
#     net.blobs['data'].data[...] = transformed_image
#     net.forward()
#     print  str(i) + str('.jpg') + '  ' + 'predicted class is:' + str(net.blobs['prob'].data[0].argmax()) + '\n'
