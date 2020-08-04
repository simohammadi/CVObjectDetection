from myhough import line_fitting
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import itertools
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split




class preproc_cable:
    TRAIN_DIR = "train_data_comp"
    VAL_DIR = "val_data_comp"
    TEST_DIR = "test_data_comp"

    labels = list()
    images = list()
    encoded_labels = list()
    train_imgs = list()
    val_imgs = list()
    train_labels = list()
    val_labels = list()
    test_imgs = list()
    test_labels = list()

    def __init__(self, dir_containing_imgs):
        #TODO save label to each file such that validation is possible
        files = glob.glob(dir_containing_imgs+"/*")
        samples = self.read_samples(files)
        seg_samples = self.seg_samples(samples)

        self.images = self.crop_spacer(samples, seg_samples)
        self.images = self.resize(self.images)
        print(files)
        filename = "labels.txt"
        if(filename not in os.listdir(os.curdir)):
            self.set_labels(self.images, filename)
        else:
            self.labels = self.read_labels(filename)
            if(len(self.labels) != len(self.images)):
                self.set_labels(self.images)

        
        self.encoded_labels = self.get_encoded(self.labels)
        [self.train_imgs, self.val_imgs, self.test_imgs, self.train_labels, self.val_labels, self.test_labels] = self.get_datasets(self.images)
        self.save_datasets(self.train_imgs, self.val_imgs, self.test_imgs, self.train_labels, self.val_labels, self.test_labels)
        self.train_imgs = self.get_images(self.train_imgs)
        self.val_imgs = self.get_images(self.val_imgs)
        self.train_labels = self.categorical_encoded(self.train_labels)
        self.val_labels = self.categorical_encoded(self.val_labels)

    def save_datasets(self, train, val, test, train_labels,val_labels, test_labels):
        os.mkdir(self.TRAIN_DIR) if not os.path.isdir(self.TRAIN_DIR) else None
        os.mkdir(self.VAL_DIR) if not os.path.isdir(self.VAL_DIR) else None
        os.mkdir(self.TEST_DIR) if not os.path.isdir(self.TEST_DIR) else None

        self.save_imgs(train, self.TRAIN_DIR)
        self.save_imgs(val, self.VAL_DIR)
        self.save_imgs(test, self.TEST_DIR)

        self.save_labels(train_labels, self.TRAIN_DIR)
        self.save_labels(val_labels, self.VAL_DIR)
        self.save_labels(test_labels, self.TEST_DIR)

    def save_labels(self, labels, path):
        filename = path + "/labels.txt"
        with open(filename, "w") as output:
            output.write(str(labels))

    def save_imgs(self, imgs, path):
        size = len(imgs)
        
        for (img, i) in zip(imgs, range(0, size)):
            fn = path + "/" + str(i) + ".jpg"
            cv.imwrite(fn, img)
            
    def read_samples(self, files):
        return [cv.imread(fn) for fn in files]

    def seg_samples(self, imgs):
        return [cv.Canny(im, 50, 200, None, 3) for im in imgs]
    
    def crop_spacer(self, samples, seg_samples):
        cropped_samples = list()
        
        for (sample, seg_sample) in zip(samples, seg_samples):
            cropped_samples.append(line_fitting(seg_sample, sample))
        
        return cropped_samples

    def get_shapes(self, images):
        shapes = np.empty((0,3), int)
        
        for img in images:
            shapes = np.vstack([shapes,img.shape])
        return shapes
    
    def filter_bad_images(self, images, shapes):
        count = 0
        size = range(0,len(images))
        for i in size:
            if(shapes[i-count][1] < 300):
                del images[i-count]
                shapes = np.delete(shapes, i-count, axis=0)
                count += 1
        return shapes, images
    
    def get_new_shapes(self, images, shapes):
        
        mean_width = int(np.mean(shapes[:,1]))
        mean_height = int(np.mean(shapes[:,0]))
        diff = abs(mean_width - mean_height)
        delta = diff/2

        if mean_height > mean_width:
            new_shape = [int(mean_height - delta), int(mean_width + delta)]
        else:
            new_shape = [int(mean_height + delta), int(mean_width - delta)]
        
        return new_shape

    def resize(self, images):
        shapes = self.get_shapes(images)
        
        [shapes, images] = self.filter_bad_images(images, shapes)
        new_shape = self.get_new_shapes(images, shapes)

        size = len(images)
        for i in range(0,size):
            images[i] = cv.resize(images[i],(new_shape[0], new_shape[1]))
        return images

    def set_labels(self, images, filename):
        size = len(images)
        
        for i in range(0, size):
            plt.imshow(images[i])
            plt.show()
            input_var = int(input())
            self.labels.append(input_var)
        
        with open(filename, "w") as output:
            output.write(str(self.labels))

    def read_labels(self, filename):
        with open(filename , "r") as read:
            labels = read.read()
        labels = str(labels)[1:-1]
        labels = labels.split(", ")

        temp = list()
        for i in range(0,len(labels)):
            temp.append(int(labels[i]))
        return temp

    def get_encoded(self, labels):
        le = LabelEncoder()
        le.fit(labels)
        encoded_label = le.transform(labels)
        return encoded_label
    
    def categorical_encoded(self, labels):
        enc_labels = to_categorical(labels)
        return enc_labels

    def get_datasets(self, images):
        train, val, train_labels, val_labels = train_test_split(images, self.encoded_labels, test_size=0.3)
        train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size = 0.2)
        return train, val, test, train_labels, val_labels, test_labels
    
    def get_images(self, images):
        imgs = [img_to_array(img) for img in images]
        return np.array(imgs)

def main():
    p = preproc_cable("/home/sina/Documents/abb/pictures/good_candidate")

if __name__ == "__main__":
    main()
