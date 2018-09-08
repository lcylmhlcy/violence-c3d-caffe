#encoding:utf-8

import os.path
import glob
#path = "./train/violence"  #此处为数据集（train和val）的绝对路径
#labellist = os.listdir(path)
#for label in labellist:
#     newpath = os.path.join(path, label)
#     # print newpath, label
#     for root, dirs, files in os.walk(newpath):
#            for file in files:
#                  print root, file
#                  f = open( "./data/train.txt", "a")
#                  f.write(os.path.join("violence",dirs, file) + " " + "1" + "\n")


path = "./train_movies/no-violence"  #此处为数据集（train和val）的绝对路径
path1 = "./examples/violence_c3d/train_movies/no-violence"
for root, dirs, files in os.walk(path):
   for file in files:
        #print root, file
        f = open( "./c3d_train.txt", "a")
        f.write(os.path.join(path1, file) + " " + "0" + "\n")



#for i in range(1,11):
#    path = "./train/violence/" + str(i) #此处为数据集（train和val）的绝对路径
#    path1 = "violence/" + str(i)
#    for root, dirs, files in os.walk(path):
#        for file in files:
#            #print root, file
#            f = open( "./data/train.txt", "a")
#            f.write(os.path.join(path1, file) + " " + "1" + "\n")