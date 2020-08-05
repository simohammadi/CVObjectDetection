import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
import model_evaluation_utils as meu
import os
from spacer_cropper import spacer_cropper 
import matplotlib.pyplot as plt
import natsort

import sys

class tester:

    model = None

    def __init__(self, path_to_model, mode, path_to_imgs, IMG_DIM, path_to_labels):
        
        num2class_label_trans = lambda l: ["trans" if x==1 else "not_trans" for x in l]
        #class2num_label_trans = lambda l: [1 if x=="trans" else 0 for x in l]
        
        self.model = load_model(path_to_model)
        
        if(mode == "trans"):
            files = self.read_files(path_to_imgs)
            labels = self.get_test_labels_trans(files)
            imgs = self.get_scaled_imgs(files, IMG_DIM)
            predictions = self.predict(imgs,path_to_model, IMG_DIM)
            predictions = num2class_label_trans(predictions)
            self.plot(labels, predictions)
        else:

            files = sorted(self.read_files(path_to_imgs))
            files = natsort.natsorted(files, reverse = False)
            imgs = self.get_scaled_imgs(files, IMG_DIM)
            labels = self.read_labels_comp(path_to_labels)
            predictions = self.predict(imgs,path_to_model, IMG_DIM)
            self.plot(labels, predictions)

    def read_labels_comp(self, filename):
        with open(filename , "r") as read:
            labels = read.read()
        labels = str(labels)[1:-1]
        labels = labels.split(" ")

        temp = list()
        for i in range(0,len(labels)):
            temp.append(int(labels[i]))
        return temp
    
    def get_test_labels_trans(self, files):
        labels = list()

        for fn in files:
            basename = os.path.basename(fn)
            strings = os.path.splitext(basename)[0].split("_")
            if(strings[0] == "trans"):
                labels.append("trans")
            else:
                labels.append("not_trans")
        return labels
    
    def read_files(self, path):
        return glob.glob(path)

    def get_scaled_imgs(self, files, IMG_DIM):
        imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in files]
        imgs = np.array(imgs)
        imgs_scaled = imgs.astype("float32")
        imgs_scaled /= 255
        return imgs_scaled

    def predict(self, imgs, path_to_model, IMG_DIM):
        base_name = os.path.basename(path_to_model)
        strings = os.path.splitext(base_name)[0].split("_")
        if strings[0] == "basic":
            img_dim = [IMG_DIM[0], IMG_DIM[1], 3]
            test_features = self.get_bottleneck_features(self.import_vgg_model(img_dim), imgs)
            return self.model.predict_classes(test_features, verbose = 0)
        else:
            return self.model.predict_classes(imgs, verbose = 0)


    def plot(self, labels, predictions):
        meu.display_model_performance_metrics(true_labels=labels, predicted_labels=predictions, classes=list(set(labels)))
        import Plot_conf_matrix as pl
        pl.plot_confusion_matrix(labels,predictions,['8','9','12','22'], normalize=True)
        plt.show()

    
    def set_labels(self, images, filename):
        size = len(images)
        labels = list()
        
        for i in range(0, size):
            plt.imshow(images[i])
            plt.show()
            input_var = int(input())
            labels.append(input_var)
        
        with open(filename, "w") as output:
            output.write(str(labels))
        return labels

    def get_bottleneck_features(self, model, images):
        return model.predict(images, verbose=0)


    def import_vgg_model(self, input_shape):

        from keras.applications import vgg16
        from keras.models import Model
        import keras

        vgg = vgg16.VGG16(include_top = False, weights = "imagenet", input_shape=input_shape)
        output = vgg.layers[-1].output
        output = keras.layers.Flatten()(output)

        #TODO find out why model is defined with vgg input and its output, seems unnecessary
        vgg_model = Model(vgg.input, output)
        vgg_model.trainable = False

        for layer in vgg_model.layers:
            layer.trainable = False
        
        return vgg_model

def main():
    mt = tester(sys.argv[1], None, sys.argv[2], (432, 432), sys.argv[3])

if __name__ == "__main__":
    main()
