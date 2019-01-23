# Code Created By: Abhishek Bhardwaj
# 23 January 2019

import numpy as np
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import glob
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2

class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="./data/train/image", label_path="./data/train/label",
                 test_path="./data/test/image", npy_path="./npydata", img_type="png"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        i=0
        print('Processing Training Images..')
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype = np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype = np.uint8)

        for x in range(len(imgs)):
            imgpath = imgs[x]
            pic_name = imgpath.split('/')[-1]
            labelpath = self.label_path + '/' + pic_name
            img = load_img(imgpath, grayscale=False, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i%2 == 0:
                print ('Done: {0}/{1} images'. format(i, len(imgs)))
            i += 1

        print('Processing done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Coded images to .npy files')

    def create_test_data(self):
        i=0
        print('Processing Test Images..')
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False, target_size=[512, 512])
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        txtname = './results/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('Test Images Processed Successfully')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print("Saved npy files to imgs_test.npy.")

    def load_train_data(self):
        print('Loading Train Images..')
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        print('Training Images Loaded')
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('=' * 25)
        print('Loading Test Images..')
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test


class loadUnet(object):
    def __init__(self, img_rows=512, img_cols=512, training=False):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.training = training

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def load_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print("Using Dropout..")
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        print("Using Dropout..")
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(input=inputs, output=conv10)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        if(self.training==False):
            model.load_weights("unet.hdf5")
        return model

    def train(self):
        print("Loading Train Matrix..")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("Loading Done")
        print("Loading Deep Neural Network(U-Net)")
        model = self.load_unet()
        print("Unet found and Loaded")
        if(self.training==True):
            model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
            print('Training Unet...')
            model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=1, verbose=1, validation_split=0, shuffle=True, callbacks=[model_checkpoint])
        else:
            print("Using Pre-Trained Weights..")
        print("Predicting Test Results")
        imgs_mask_test = model.predict(imgs_test, batch_size=2, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("Converting Test array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        piclist = []
        for line in open("./results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        get_size = cv2.imread('./data/test/image/' + piclist[0]);
        height, width, channel = get_size.shape
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = imgs[i]
            img = array_to_img(img)
            img.save(path)
            cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv_pic = cv2.resize(cv_pic, (width, height), interpolation=cv2.INTER_CUBIC)
            binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, cv_save)
        print("Coversion Complete")

if __name__ == '__main__':
    Data = dataProcess(512,512)
    Data.create_train_data()
    Data.create_test_data()
    Unet = loadUnet()
    Unet_model = Unet.load_unet()
    Unet.train()
    Unet.save_img()
    print("Process Completed Successfully")
