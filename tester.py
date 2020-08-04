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

class tester:

    model = None
    #TODO pass labels from preproc_cable or use labeling in filename
    def __init__(self, path, mode, path_to_imgs, IMG_DIM):
        
        num2class_label_trans = lambda l: ["trans" if x==1 else "not_trans" for x in l]
        class2num_label_trans = lambda l: [1 if x=="trans" else 0 for x in l]
        
        self.model = load_model(path)
        
        if(mode == "trans"):
            files = self.read_files(path_to_imgs)
            labels = self.get_test_labels_trans(files)
            imgs = self.get_scaled_imgs(files, IMG_DIM)
            predictions = self.predict(imgs)
            predictions = num2class_label_trans(predictions)
            self.plot(labels, predictions)
        else:
            #TODO Write the testing part for cable detection
            files = sorted(self.read_files(path_to_imgs))
            files = natsort.natsorted(files, reverse = False)
            imgs = self.get_scaled_imgs(files, IMG_DIM)
            labels = self.read_labels_comp()
            predictions = self.predict(imgs)
            self.plot(labels, predictions)
    #TODO fix such that you dont hav to change filename
    def read_labels_comp(self):
        filename = "test_data_comp/labels.txt"
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

    def predict(self, imgs):
        return self.model.predict_classes(imgs, verbose = 0)

    def plot(self, labels, predictions):
        meu.display_model_performance_metrics(true_labels=labels, predicted_labels=predictions, classes=list(set(labels)))
    
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


def main():
    #t = tester("finetuned.h5", "trans", "test_data/*", (150,150))
    t = tester("finetuned_cablecable.h5", None, "test_data_comp/*.jpg", (460, 460))
if __name__ == "__main__":
    main()