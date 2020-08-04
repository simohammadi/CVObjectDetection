import glob
import numpy as np
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder


class preproc_transition:

    TRAIN_DIR = "training_data"
    VAL_DIR = "validation_data"
    TEST_DIR = "test_data"
    IMG_DIM = (150,150)
    TRANS_DIR = "/transition"
    NON_TRANS_DIR = "/non_trans"

    encoded_train_label = list()
    encoded_val_label = list()
    train_files = list()
    train_imgs = list()
    validation_imgs = list()
    test_imgs = list()

    #possibly change the paths to inputs
    def __init__ (self, trans_files_path, non_trans_files_path):
        trans_files = glob.glob(trans_files_path)
        non_trans_files = glob.glob(non_trans_files_path)
        
        edge_non_trans_files = self.find_edge_files(non_trans_files)
        edge_trans_files = self.find_edge_files(trans_files)
        
        [train_non_trans, validation_non_trans, test_non_trans] = self.get_datasets(edge_non_trans_files)
        [train_trans, validation_trans, test_trans] = self.get_datasets(edge_trans_files)
        
        self.make_directory(train_trans, validation_trans, test_trans, train_non_trans, validation_non_trans, test_non_trans)

        #Until now is just to get files in correct folder with labeled naming schema
        self.train_imgs = self.get_images(self.TRAIN_DIR)
        self.validation_imgs = self.get_images(self.VAL_DIR)


        self.train_files = glob.glob(self.TRAIN_DIR+"/*")
        train_labels = self.get_labels(self.train_files)
        val_labels = self.get_labels(glob.glob(self.VAL_DIR+"/*"))

        self.encoded_train_label = self.get_encoded(train_labels)
        self.encoded_val_label = self.get_encoded(val_labels)


    def find_edge_files(self, files):
        path_list = list()
        for fn in files:
            if("bot" in os.path.basename(fn) or "top" in os.path.basename(fn)):
                path_list.append(fn)
        return path_list
    
    def get_partition(self, files):
        return np.random.choice(files, size = int(len(files)/2), replace = False)
    
    #quite sure this is unecessary
    def get_test_set(self, files):
        return np.random.choice(files, size = len(files), replace = False)

    def print_shape(self, train, validation, test):
        print("datasets shapes", train.shape, validation.shape, test.shape)
    
    def get_datasets(self, files):
        train = self.get_partition(files)
        files = list(set(files)-set(train))
        validation = self.get_partition(files)
        files = list(set(files)-set(validation))
        test = self.get_test_set(files)
        return [train, validation, test]
    
    def make_directory(self, train_trans, validation_trans, test_trans, train_non_trans, validation_non_trans, test_non_trans):
        #TODO check this
        self.clean_directory()

        os.mkdir(self.TRAIN_DIR) if not os.path.isdir(self.TRAIN_DIR) else None
        os.mkdir(self.VAL_DIR) if not os.path.isdir(self.VAL_DIR) else None
        
        for fn in train_trans:
            old_name = shutil.copy(fn, self.TRAIN_DIR)
            basename = os.path.basename(old_name)
            new_basename = "trans_" + basename
            name_path = self.TRAIN_DIR + "/"+ new_basename
            os.rename(old_name, name_path)

        for fn in validation_trans:
            old_name = shutil.copy(fn, self.VAL_DIR)
            basename = os.path.basename(old_name)
            new_basename = "trans_" + basename
            name_path = self.VAL_DIR + "/"+ new_basename
            os.rename(old_name, name_path)
        
        for fn in test_trans:
            old_name = shutil.copy(fn, self.TEST_DIR)
            basename = os.path.basename(old_name)
            new_basename = "trans_" + basename
            name_path = self.TEST_DIR + "/"+ new_basename
            os.rename(old_name, name_path)
        
        for fn in train_non_trans:
            shutil.copy(fn, self.TRAIN_DIR)
        
        for fn in validation_non_trans:
            shutil.copy(fn, self.VAL_DIR)
        
        for fn in test_non_trans:
            shutil.copy(fn, self.TEST_DIR)

    def get_images(self, path):
        files = glob.glob(path+"/*")
        imgs = [img_to_array(load_img(img, target_size=self.IMG_DIM)) for img in files]
        return np.array(imgs)


    def remove_contents(self, path):
        filelist = [f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f)) #if not os.path.isdir(f) else None
    
    def clean_directory(self):
        self.remove_contents(self.TRAIN_DIR)
        self.remove_contents(self.VAL_DIR)
        self.remove_contents(self.TEST_DIR)

    def get_labels(self, files):
        labels = list()
        for fn in files:
            if "trans" in fn:
                labels.append("trans")
            else:
                labels.append("non-trans")
        return labels

    def get_encoded(self, labels):
        le = LabelEncoder()
        le.fit(labels)
        encoded_label = le.transform(labels)
        return encoded_label
    

    #Objective
    #get_trainset, get_valset, get_testset, get_labels (for train and val)
     
def main():
    transfiles_path = "/home/sina/Documents/abb/patches/with_transitions/*"
    non_transfiles_path = "/home/sina/Documents/abb/patches/no_transition/*"

    p = preproc_transition(transfiles_path, non_transfiles_path)
    print(len(p.encoded_train_label))
    
if __name__ == "__main__":
    main()
