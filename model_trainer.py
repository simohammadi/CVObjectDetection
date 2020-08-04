import glob
import numpy as np
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import InputLayer

import pickle

class model_trainer:

    history = None
    batch_size = 100
    epochs = 10
    steps_per_epoch = 100
    total_classes = None

    def __init__(self, mode, train, val, train_label, val_label, input_shape, model_name, multiclass):

        args = train, train_label, val, val_label, input_shape, model_name
        
        if(multiclass):
            self.total_classes = train_label.shape[1]
        
        if mode == "plain_cnn":
            self.history = self.plain_cnn(*args)
        elif mode == "deep_augmented_cnn":
            self.history = self.deep_augmented_cnn(*args)
        elif mode == "basic_transferlearning":
            self.history = self.basic_transferlearning(*args)
        elif mode == "augmented_transferlearning":
            self.history = self.augmented_transferlearning(*args)
        elif mode == "finetune_transferlearning":
            self.history = self.finetune_transferlearning(*args)

        self.save_history(self.history, mode, model_name)

    def save_history(self, history, mode, model_name):
        os.mkdir("history") if not os.path.isdir("history") else None
        
        with open("history/" + mode + "_" + model_name + "_history" , "wb") as file_pi:
            pickle.dump(history, file_pi)
    
    def load_history(self, path):
        return pickle.load(open(path, "rb"))

    def plain_cnn(self, train, train_label, val, val_label, input_shape, model_name):
        
        train_generator = self.scale_data(train, train_label)
        val_generator = self.scale_data(val, val_label)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        
        if(self.total_classes):
            model.add(Dense(self.total_classes, activation="softmax"))
        else:
            model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(),
                    metrics=['accuracy'])

        model.summary()

        model.save("plain_cnn_" + model_name + ".h5")     

        return model.fit(train_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                                    validation_data=val_generator, validation_steps=50, verbose=1)

    def deep_augmented_cnn(self, train, train_label, val, val_label, input_shape, model_name):
        
        [train_generator, val_generator] = self.augment_data(train, val, train_label, val_label)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        
        if(self.total_classes):
            model.add(Dense(self.total_classes, activation="softmax"))
        else:
            model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-4),
                    metrics=['accuracy'])

            
        history = model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs*2,
                                    validation_data=val_generator, validation_steps=50, verbose=1)

        model.save("deep_augmented_cnn_" + model_name + ".h5")      
        return history

    def basic_transferlearning(self, train, train_label, val, val_label, input_shape, model_name):

        vgg_model = self.import_vgg_model(input_shape)

        train_features = self.get_bottleneck_features(vgg_model, self.scale_bottleneck(train))
        val_features = self.get_bottleneck_features(vgg_model, self.scale_bottleneck(val))

        input_shape = vgg_model.output_shape[1]

        model = Sequential()
        model.add(InputLayer(input_shape=(input_shape,)))
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        
        if(self.total_classes):
            model.add(Dense(self.total_classes, activation="softmax"))
        else:
            model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-4),
                    metrics=['accuracy'])

        model.summary()
        history = model.fit(x=train_features, y=train_label,
                        validation_data=(val_features, val_label),
                        batch_size=self.batch_size,
                        epochs=100,
                        verbose=1)
        model.save("basic_transferlearning_"+ model_name + ".h5")
        return history
    
    def augmented_transferlearning(self, train, train_label, val, val_label, input_shape, model_name):

        [train_generator, val_generator] = self.augment_data(train_imgs=train, val_imgs=val,train_label=train_label,val_label = val_label)
        vgg_model = self.import_vgg_model(input_shape)

        model = Sequential()
        model.add(vgg_model)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        
        if(self.total_classes):
            model.add(Dense(self.total_classes, activation="softmax"))
        else:
            model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=2e-5),
                    metrics=['accuracy'])
                    
        history = model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs*2,
                                    validation_data=val_generator, validation_steps=50, 
                                    verbose=1)
        model.save("augmented_tf_" + model_name + ".h5")
        return history

    def finetune_transferlearning(self, train, train_label, val, val_label, input_shape, model_name):
        
        [train_generator, val_generator] = self.augment_data(train_imgs=train, val_imgs=val,train_label=train_label,val_label = val_label)
        
        vgg_model = self.import_vgg_model(input_shape)
        vgg_model = self.set_vgg_trainable(vgg_model)


        model = Sequential()
        model.add(vgg_model)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        
        if(self.total_classes):
            model.add(Dense(self.total_classes, activation="softmax"))
        else:
            model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-5),
                    metrics=['accuracy'])
                    
        history = model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch, epochs=5,
                                    validation_data=val_generator, validation_steps=50, 
                                    verbose=1)
        model.save("finetuned_" + model_name + ".h5")
        return history       

    def import_vgg_model(self, input_shape):
        #TODO find out what this section is actually doing

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

    def scale_bottleneck(self, imgs):
        imgs = np.array(imgs)
        imgs_scaled = imgs.astype("float32")
        imgs_scaled /= 255

        return imgs_scaled
    
    def scale_data(self, imgs, labels):
        data = ImageDataGenerator(rescale=1./255)
        generator = data.flow(imgs, labels, batch_size=30)
        
        return generator

    def augment_data(self, train_imgs, val_imgs, train_label, val_label):
        
        train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                    horizontal_flip=True, fill_mode="nearest")
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow(train_imgs, train_label, batch_size=30)
        val_generator = val_datagen.flow(val_imgs, val_label, batch_size=30)
        return train_generator, val_generator

    def set_vgg_trainable(self, vgg_model):
        vgg_model.trainable = True

        set_trainable = False
        for layer in vgg_model.layers:
            if layer.name in ["block5_conv1", "block4_conv1"]:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        return vgg_model


    def get_bottleneck_features(self, model, images):
        return model.predict(images, verbose=0)

    def plot(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    def visualize_vgg_trainable(self, vgg_model):
        pd.set_option("max_colwidth", 1)
        layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
        df = pd.DataFrame(layers, columns=["Layer Type", "Layer Name", "Layer Trainable"])
        print(df)
    
    def set_batch_size(self, new_size):
        self.batch = new_size
        
    def set_epochs(self, new_size):
       self.epochs = new_size
    
    def set_steps_per_epoch(self, steps):
        self.steps_per_epoch = steps

def main():
    import preproc_cable as pc
    import preproc_transition as pt
    
    modes = ["basic_transferlearning", "augmented_transferlearning", "finetune_transferlearning"]
    """
    p_cable = pc.preproc_cable("/home/sina/Documents/abb/pictures/good_candidate")
    
    #settings
    input_shape = [460, 460, 3]
    model_name = "cable"
    #mt = model_trainer(mode, p_cable.train_imgs, p_cable.val_imgs, p_cable.train_labels, p_cable.val_labels, input_shape, model_name, True)

    for mode in modes:
        mt = model_trainer(mode, p_cable.train_imgs, p_cable.val_imgs, p_cable.train_labels, p_cable.val_labels, input_shape, model_name, True)
    """

    
    transfiles_path = "/home/sina/Documents/abb/refined_data/patches/trans/*"
    non_transfiles_path = "/home/sina/Documents/abb/refined_data/patches/non_trans/*"

    p_trans = pt.preproc_transition(transfiles_path, non_transfiles_path)

    
    #settings
    input_shape = [150,150,3]
    model_name = "trans"


    for mode in modes:
        mt = model_trainer(mode, p_trans.train_imgs, p_trans.validation_imgs, p_trans.encoded_train_label, p_trans.encoded_val_label, input_shape, model_name, False)
    """
    for mode in modes:
        history = mt.load_history("history/" + mode + "_cable" + "_history")
        mt.plot(history)
    """

if __name__ == "__main__":
    main()