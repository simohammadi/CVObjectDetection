#Modules
from model_trainer import model_trainer 
from preproc_cable import preproc_cable
from preproc_transition import preproc_transition
from patcher import patcher
from spacer_cropper import spacer_cropper
from tester import tester
from predictor import predictor

import os
from keras.models import load_model
import cv2 as cv

class ai_model:

    def __init__(self, mode):
        path_trans = "/home/sina/Documents/abb/patches/with_transitions/*"
        path_non_trans = "/home/sina/Documents/abb/patches/no_transition/*"
        path_img_trans = "/home/sina/Documents/abb/transition_pictures/with_spacer/good_candidate/DSC_4762.jpg"
        #self.transition_trainer(path_trans, path_non_trans)
        #self.tester("finetuned.h5", "test_data/*", (150,150), "trans")
        #self.tester("finetuned_cablecable.h5", "train_data_comp/*.jpg", (460, 460), None)
        #self.classify("finetuned.h5", path_img_trans, 150, "trans")
        #self.classify("finetuned_cablecable.h5", path_img_trans, 460, None)

    
    def trainer(self, mode):
        NotImplemented
        #TODO write trainer
    
    def transition_trainer(self, path_trans_files, non_trans_files_path):
        dir_name = "trans_patches"
        patch_dim = [150,150]
        patcher_trans = patcher(path_trans_files, patch_dim, dir_name, gen_data=True)
        dir_name = "non_trans_patches"
        patcher_non_trans = patcher(non_trans_files_path, patch_dim, dir_name, gen_data=True)
        preproc = preproc_transition(path_trans_files, non_trans_files_path)
        mode = "finetune_transferlearning"
        name = "test"
        input_shape = patch_dim + [3]
        model = model_trainer(mode, preproc.train_imgs, preproc.validation_imgs, preproc.encoded_train_label, preproc.encoded_val_label, input_shape, name)
        model.plot(model.history)
    
    #TODO component trainer needs softmax function and so on
    def component_trainer(self, dir_files):
        preproc = preproc_cable(dir_files)
        input_shape = preproc.images[0].shape
        mode = "finetune_transferlearning"
        model = model_trainer()
    
    #TODO fix object oriented code
    def tester(self,path_to_model, path_to_imgs, img_dim, mode):
        if(mode == "trans"):
            self.test_trans(path_to_model, path_to_imgs, img_dim)
        else:
            self.test_comp(path_to_model, path_to_imgs, img_dim)
    
    #TODO check is path_to_imgs always the same
    def test_trans(self, path_to_model, path_to_imgs, patch_dim):
        t = tester(path_to_model, "trans", path_to_imgs, patch_dim)
    
    def test_comp(self, path_to_model, path_to_imgs, img_dim):
        t = tester(path_to_model, None, path_to_imgs, img_dim)

    def classify(self, model_path, img_path, img_dim, mode):
        p = predictor(model_path, img_path, img_dim, mode)
        print(p.res)
         
        
def main():
    am = ai_model("test")

if __name__ == "__main__":
    main()

