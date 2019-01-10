import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os, shutil


path = r"C:\Users\NITUK\Downloads\dataset"
path1 = r"C:\Users\NITUK\Downloads\dataset\cat"
path2 = r"C:\Users\NITUK\Downloads\dataset\dog"
base_dir = r"C:\Users\NITUK\Downloads\dataset\dog_cat_small"
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir,'val_data')
os.mkdir(val_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
train_dir_cats = os.path.join(train_dir, 'cats')
os.mkdir(train_dir_cats)
train_dir_dogs = os.path.join(train_dir, 'dogs')
os.mkdir(train_dir_dogs)
test_dir_cats = os.path.join(test_dir, 'cats')
os.mkdir(test_dir_cats)
test_dir_dogs = os.path.join(test_dir, 'dogs')
os.mkdir(test_dir_dogs)
val_dir_cat = os.path.join(val_dir, 'cat')
os.mkdir(val_dir_cat)
val_dir_dog = os.path.join(val_dir, 'dog')
os.mkdir(val_dir_dog)

fnames = ['cat{}.jpg'.format(i) for i in range(10000)]

for fname in fnames:
    scr = os.path.join(path1, fname)
    dst = os.path.join(train_dir_cats, fname)
    shutil.copy(scr,dst)

fnames = ['dog{}.jpg'.format(i) for i in range(10000)]
for fname in fnames:
    scr = os.path.join(path2, fname)
    dst = os.path.join(train_dir_dogs, fname)
    shutil.copy(scr,dst)

fnames = ['cat{}.jpg'.format(i) for i in range(10000,10500)]

for fname in fnames:
    scr = os.path.join(path1, fname)
    dst = os.path.join(test_dir_cats, fname)
    shutil.copy(scr,dst)

fnames = ['cat{}.jpg'.format(i) for i in range(10500, 11000)]

for fname in fnames:
    scr = os.path.join(path1, fname)
    dst = os.path.join(val_dir_cat, fname)
    shutil.copy(scr,dst)

fnames = ['dog{}.jpg'.format(i) for i in range(10000,10500)]
for fname in fnames:
    scr = os.path.join(path2, fname)
    dst = os.path.join(test_dir_dogs, fname)
    shutil.copy(scr,dst)

fnames = ['dog{}.jpg'.format(i) for i in range(10500,11000)]
for fname in fnames:
    scr = os.path.join(path2, fname)
    dst = os.path.join(val_dir_dog, fname)
    shutil.copy(scr,dst)

print('total no of cat images train:',len(os.listdir(train_dir_cats)))
print('total no of dog images train:',len(os.listdir(train_dir_dogs)))
print('total no of cat images test:',len(os.listdir(test_dir_cats)))
print('total no of dog images test:',len(os.listdir(test_dir_dogs)))
print('total no of cat images valid:',len(os.listdir(val_dir_cat)))
print('total no of dog images valid:',len(os.listdir(val_dir_dog)))