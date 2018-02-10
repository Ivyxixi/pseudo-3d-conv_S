import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy

from p3d_model import P3D199
from p3d_model_offset_fyq import get_P3D_offset_fyq
from p3d_model_offset_T_fyq import get_P3D_offset_T_fyq
from UCF101_fyq import UCF101


def train(dateset,model,model_name):
    myUCF101=dateset
    #loss and optimizer
    criterion=nn.CrossEntropyLoss()
    # optimizer=optim.SGD(model.parameters(),lr=0.001)
    # optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    optimizer=optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)
 
     #train the network
    for epoch in range(4):         #loop over the dataset multiple times
        running_loss=0
        batch_num=myUCF101.set_mode('train')
        for batch_index in range(batch_num):
            # get the train data
            train_x,train_y=myUCF101[batch_index]   
            # warp them in Variable
            # train_x,train_y=Variable(train_x.cuda()),Variable(train_y.type(torch.LongTensor).cuda())
            train_x,train_y=Variable(train_x).cuda(),Variable(train_y.type(torch.LongTensor)).cuda()
            # set 0
            optimizer.zero_grad()
            # forward+backwar+optimize
            out=model(train_x)
            loss=criterion(out,train_y)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss+=loss.data[0]
            print ('[%d,%d] loss: %.3f' %(epoch+1,batch_index+1,running_loss))
            print >> f, ('[%d,%d] loss: %.3f' %(epoch+1,batch_index+1,running_loss))
            running_loss=0.0
 
    torch.save(model.state_dict(), model_name)
    print ('Finished Training')
    print >> f,('Finished Training')


def test(dateset,model,model_state_path):
    myUCF101=dateset
    model.load_state_dict(torch.load(model_state_path))
    classNames=myUCF101.get_className()
    # test the network on the test data
    correct=0
    total=0
    batch_num=myUCF101.set_mode('test')
    for batch_index in range(batch_num):
        batch_correct=0
        # get the test dat
        test_x,test_y_label=myUCF101[batch_index]
        # warp teat_x in Variable
        test_x=Variable(test_x.cuda())
        # get teh predicted output
        out=model(test_x)
        _,predicted_y=torch.max(out.data,1)
        predicted_y=predicted_y.cpu().numpy()
        predicted_y=np.array(predicted_y,dtype=np.uint8)
        classNames=np.array(classNames)
        predicted_label=classNames[predicted_y]
        batch_correct+= (predicted_label==test_y_label).sum()
        print('bactch: %d  accuracy is: %.2f' %(batch_index+1,batch_correct/float(len(test_y_label))))
        print >> f, ('bactch: %d  accuracy is: %.2f' %(batch_index+1,batch_correct/float(len(test_y_label))))
        correct+=batch_correct
        total+=len(test_y_label)
    print ('Test Finished')
    print >> f, ('Test Finished')
    print ('accuracy is: %.2f' %(correct/float(total)))
    print >> f, ('accuracy is: %.2f' %(correct/float(total)))



def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys() :
        if ((not k in wf) | (k=='fc.weight') | (k=='fc.bias')):      
            wf[k] = wt[k]
    model_to.load_state_dict(wf)




if __name__ == '__main__':
    # redirect out to result_fyq.txt
    f=open('result_fyq.txt','w+') 

    #dataset
    myUCF101=UCF101()

    '''
    #model
    model = P3D199(pretrained=False,num_classes=101)
    model = model.cuda()
    # print (model)

    # model_init=P3D199(pretrained=True,num_classes=400)
    # transfer_weights(model_init,model)
    # train(myUCF101,model)

    test(myUCF101,model,'result4_model.pkl')
    '''

    # model=get_P3D_offset_fyq()
    # model=model.cuda()
    # print model
    
    # model_init=P3D199(pretrained=True,num_classes=400)
    # transfer_weights(model_init,model)

    # model.load_state_dict(torch.load('result5_model_s.pkl'))
    # train(myUCF101,model)
    
 

    model_init=P3D199(pretrained=True,num_classes=400)

    model_1 = P3D199(pretrained=False,num_classes=101)
    model_1=model_1.cuda()
    # transfer_weights(model_init,model_1)
    # train(myUCF101,model_1,'p3d.pkl')

    # model_2=get_P3D_offset_T_fyq()
    # model_2=model_2.cuda()
    # transfer_weights(model_init,model_2)
    # train(myUCF101,model_2,'p3d_t.pkl')

    test(myUCF101,model_1,'p3d.pkl')
    
    # test(myUCF101,model_1,'p3d.pkl')
