import os
import sys
import random
import numpy as np
from tqdm import tqdm

from skimage.io import imread, imshow, imsave

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

def generate_train_val_test_ids(img_path):

    train_val_ids = next(os.walk(img_path + 'train_val'))[2]
    random.seed(1)
    random.shuffle(train_val_ids)
    num_train_val = len(train_val_ids)
    print('Total number of training and validation: ' + str(num_train_val))
    
    test_ids = next(os.walk(img_path + 'test'))[2]
    test_ids.sort()
    num_test = len(test_ids)
    print('Total number of test: ' + str(num_test))
    
    val_split = 0.2
    split_cutoff = int(num_train_val*(1 - val_split))
    print('Training set = ' + str(split_cutoff) + ' and Validation set = ' + str(num_train_val - split_cutoff))    
    
    return train_val_ids, test_ids, split_cutoff

def get_image_mask_data(img_path, mask_path, train_val_ids, test_ids, IMG_WIDTH=512, IMG_HEIGHT=512, IMG_CHANNELS=3):
    
    # Num of train val test
    num_train_val = len(train_val_ids)
    num_test = len(test_ids)
    
    # Get and resize training and validation set images and masks
    X_train = np.zeros([num_train_val, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=np.float32)
    Y_train = np.zeros([num_train_val, IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.bool)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_val_ids), total=num_train_val):    
        img = imread(img_path + 'train_val/' + id_) / 255.0    
        X_train[n] = img
        mask = imread(mask_path + 'train_val/' + id_.split(".")[0] + '.png')       
        binary = mask > 0    
        Y_train[n] = np.expand_dims(binary,axis=2)    
        
            
    # Get and resize test set images and masks
    X_test = np.zeros([num_test, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=np.float32)
    Y_test = np.zeros([num_test, IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.bool)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=num_test):    
        img = imread(img_path + 'test/' + id_) / 255.0    
        X_test[n] = img
        mask = imread(mask_path + 'test/' + id_.split(".")[0] + '.png')       
        binary = mask > 0    
        Y_test[n] = np.expand_dims(binary,axis=2)     
        
    return X_train, Y_train, X_test, Y_test

def get_data_generator(X_train, Y_train, split_cutoff, batch_size=4, seed=42):
    
    # Creating the training Image and Mask generator
    image_datagen = image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range = 90, fill_mode = 'constant', cval = 0)
    mask_datagen = image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range = 90, fill_mode = 'constant', cval = 0)

    # Keep the same seed for image and mask generators so they fit together
    # Creating the training Image and Mask generator
    image_datagen.fit(X_train[:split_cutoff], augment=True, seed=seed)
    mask_datagen.fit(Y_train[:split_cutoff], augment=True, seed=seed)

    x=image_datagen.flow(X_train[:split_cutoff],batch_size=batch_size,shuffle=True, seed=seed)
    y=mask_datagen.flow(Y_train[:split_cutoff],batch_size=batch_size,shuffle=True, seed=seed)

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(X_train[split_cutoff:], augment=True, seed=seed)
    mask_datagen_val.fit(Y_train[split_cutoff:], augment=True, seed=seed)

    x_val=image_datagen_val.flow(X_train[split_cutoff:],batch_size=batch_size,shuffle=True, seed=seed)
    y_val=mask_datagen_val.flow(Y_train[split_cutoff:],batch_size=batch_size,shuffle=True, seed=seed)

    #creating a training and validation generator that generate masks and images
    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)    
    
    return train_generator, val_generator
    