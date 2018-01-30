import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import torch


# initial handle
'''
python utils_fyq/video_jpg_ucf101_hmdb51.py /home/hl/Desktop/lovelyqian/CV_Learning/UCF101 /home/hl/Desktop/lovelyqian/CV_Learning/UCF101_jpg
python utils_fyq/n_frames_ucf101_hmdb51.py /home/hl/Desktop/lovelyqian/CV_Learning/UCF101_jpg`
python utils_fyq/ucf101_json.py /home/hl/Desktop/lovelyqian/CV_Learning/UCF101_TrainTestlist 
'''

class UCF101:
    def __init__(self,mode='train'):
        self.videos_path='/home/hl/Desktop/lovelyqian/CV_Learning/UCF101_jpg'
        self.csv_dir_path='/home/hl/Desktop/lovelyqian/CV_Learning/UCF101_TrainTestlist/'
        self.label_csv_path = os.path.join(self.csv_dir_path, 'classInd.txt')
        self.transform = transforms.Compose([
                transforms.RandomCrop(160),            #size
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])         #nedd to calculute
                ])
        # self.batch_size=128
        self.batch_size=8
        self.mode= mode

        self.get_train()
        self.get_test()

        
    def get_className(self):
        data = pd.read_csv(self.label_csv_path, delimiter=' ', header=None)
        labels = []
        # labels.append("0")
        for i in range(data.shape[0]):
            labels.append(data.ix[i, 1])
        return labels

    def get_train(self):
        train_x_path = []
        train_y = []
        for index in range(1,4):
            tmp_path='trainlist0'+str(index)+'.txt'
            train_csv_path = os.path.join(self.csv_dir_path, tmp_path)
            # print (train_csv_path)

            data = pd.read_csv(train_csv_path, delimiter=' ', header=None)
            for i in range(data.shape[0]):
                train_x_path.append(data.ix[i,0])
                # train_y.append(data.ix[i,1])
                train_y.append(data.ix[i,1]-1)
    
        self.train_num=len(train_x_path)
        self.train_x_path=train_x_path
        self.train_y=train_y
        return train_x_path,train_y


    def get_test(self):
        test_x_path=[]
        test_y_label=[]
        for index in range(1,4):
            temp_path='testlist0'+str(index)+'.txt'
            test_csv_path=os.path.join(self.csv_dir_path,temp_path)
            # print (test_csv_path)

            data=pd.read_csv(test_csv_path,delimiter=' ',header=None)
            for i in range(data.shape[0]):
                test_x_path.append(data.ix[i,0])
                label=self.get_label(data.ix[i,0])
                test_y_label.append(label)
        self.test_num=len(test_x_path)
        self.test_x_path=test_x_path
        self.test_y_label=test_y_label
        return test_x_path,test_y_label


    def get_label(self,video_path):
        slash_rows = video_path.split('/')
        class_name = slash_rows[0]
        return class_name

    def get_single_image(self,image_path):
        img=Image.open(image_path)
        transformed_img=self.transform(img)
        img.close()
        return transformed_img       

    def get_single_image_2(self,image_path):
        image=resize(io.imread(image_path),output_shape=(160,160),preserve_range= True)    #240,320,3--160,160,3
        # io.imshow(image.astype(np.uint8))
        # io.show()
        image =image.transpose(2, 0, 1)              #3,160,160
        return torch.from_numpy(image)               #range[0,255]

    def get_single_video_x(self,train_x_path):
        slash_rows=train_x_path.split('.')
        dir_name=slash_rows[0]
        video_jpgs_path=os.path.join(self.videos_path,dir_name)
        ##get the random 16 frame
        data=pd.read_csv(os.path.join(video_jpgs_path,'n_frames'),delimiter=' ',header=None)
        frame_count=data[0][0]
        train_x=torch.Tensor(16,3,160,160)

        image_start=random.randint(1,frame_count-17)
        image_id=image_start
        for i in range(16):
            s="%05d" % image_id
            image_name='image_'+s+'.jpg'
            image_path=os.path.join(video_jpgs_path,image_name)
            # single_image=self.get_single_image(image_path)
            single_image=self.get_single_image_2(image_path)
            train_x[i,:,:,:]=single_image
            image_id+=1
        return train_x
        
        '''
        intervar=int(frame_count/16)
        image_id=1-intervar
        for i in range(16):
            image_id+=intervar
            s="%05d" % image_id
            image_name='image_'+s+'.jpg'
            image_path=os.path.join(video_jpgs_path,image_name)
            # single_image=self.get_single_image(image_path)
            single_image=self.get_single_image_2(image_path)
            train_x[i,:,:,:]=single_image
        return train_x
        '''
    
    def get_minibatches_index(self, shuffle=True):
        """
        :param n: len of data
        :param minibatch_size: minibatch size of data
        :param shuffle: shuffle the data
        :return: len of minibatches and minibatches
        """
        if self.mode=='train':
            n=self.train_num
        elif self.mode=='test':
            n=self.test_num

        minibatch_size=self.batch_size
        
        index_list = np.arange(n, dtype="int32")
 
        # shuffle
        if shuffle:
            random.shuffle(index_list)
 
        # segment
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(index_list[minibatch_start:minibatch_start + minibatch_size])
            minibatch_start += minibatch_size
 
        # processing the last batch
        if (minibatch_start != n):
            minibatches.append(index_list[minibatch_start:])
        
        if self.mode=='train':
            self.minibatches_train=minibatches
        elif self.mode=='test':
            self.minibatches_test=minibatches
        return 


    
    def __getitem__(self, index):
        if self.mode=='train':
            batches=self.minibatches_train[index]
            N=batches.shape[0]
            train_x=torch.Tensor(N,16,3,160,160)
            train_y=torch.Tensor(N)
            for i in range (N):
                tmp_index=batches[i]
                tmp_video_path=self.train_x_path[tmp_index]
                tmp_train_x= self.get_single_video_x(tmp_video_path)
                tmp_train_y=self.train_y[tmp_index]
                train_x[i,:,:,:]=tmp_train_x
                train_y[i]=tmp_train_y
            train_x=train_x.permute(0,2,1,3,4)
            return train_x,train_y
        elif self.mode=='test':
            batches=self.minibatches_test[index]
            N=batches.shape[0]
            test_x=torch.Tensor(N,16,3,160,160)
            test_y_label=[]
            for i in range (N):
                tmp_index=batches[i]
                tmp_video_path=self.test_x_path[tmp_index]
                tmp_test_x= self.get_single_video_x(tmp_video_path)
                tmp_test_y=self.test_y_label[tmp_index]
                test_x[i,:,:,:]=tmp_test_x
                test_y_label.append(tmp_test_y)
            test_x=test_x.permute(0,2,1,3,4)
            return test_x,test_y_label
    
    def set_mode(self,mode):
        self.mode=mode
        if mode=='train':
            self.get_minibatches_index()
            return self.train_num // self.batch_size
        elif mode=='test':
            self.get_minibatches_index()
            return self.test_num // self.batch_size





##  usage 

if __name__=="__main__":
    myUCF101=UCF101()
    image=myUCF101.get_single_image_2('/home/hl/Desktop/lovelyqian/CV_Learning/UCF101_jpg/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/image_00001.jpg')
    print (image)
    image2=myUCF101.get_single_image('/home/hl/Desktop/lovelyqian/CV_Learning/UCF101_jpg/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/image_00001.jpg')
    print image2

    myUCF101.get_single_video_x('/home/hl/Desktop/lovelyqian/CV_Learning/UCF101_jpg/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01')
    className=myUCF101.get_className()

    # print (className)

    # train_x_path,train_y=myUCF101.get_train()
    # for i in range (len(train_x_path)):
        # print(train_x_path[i],train_y[i],className[train_y[i]])
    # print len(train_x_path)

    # test_x_path,test_y_label=myUCF101.get_test()
    # for i in range (len(test_x_path)):
        # print(test_x_path[i],test_y_label[i])
    # print len(test_x_path)

    # myUCF101.get_single_image('h')

    # train_x_path,train_y=myUCF101.get_train()
    # print (len(train_x_path))

    #     n=5
    #     for i in range(n):
    #         train_x,train_y=myUCF101[i]
    #         print (train_x,train_y)

    
    # train
    batch_num=myUCF101.set_mode('train')
    for batch_index in range(batch_num):
        train_x,train_y=myUCF101[batch_index]
        print (train_x,train_y)
        print ("train batch:",batch_index)
    
    #TEST
    batch_num=myUCF101.set_mode('test')
    for batch_index in range(batch_num):
        test_x,test_y_label=myUCF101[batch_index]
        print test_x,test_y_label
        print ("test batch: " ,batch_index)
