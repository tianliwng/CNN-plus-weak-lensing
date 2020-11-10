from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import time
import os

######################## define constants etc. ############################### 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

rootpath = '/n/holyscratch01/dvorkin_lab/Users/tianliwang/maps_unzipped/'  # location of maps 

# these parameters are defined in the paper 
n_epoch = 30     # number of epochs to run 
n_imagespercosmo = 512  # number of images per cosmology 
n_perbatch = 32         # batch size for training 
n_perbatch_validate = n_perbatch  # set it to the same as training for now 
n_batch = n_imagespercosmo/n_perbatch  # number of minibatches 
dim_image = 1024 
dim_downsized = int(dim_image/2)            # downsized image dimension 

trainFraction = 10.0/16  # fraction of images per cosmology for training 
validateFraction = 3.0/16
testFraction = 1-trainFraction-validateFraction

n_cosmosToTrain = 90 
n_trainimages = int(n_imagespercosmo*trainFraction)    # number of images per cosmology to train
n_validateimages = int(n_imagespercosmo*validateFraction)   # number of images per cosmology to validate
n_batch_train = int(n_trainimages/n_perbatch)  # number of minibatches per cosmology to train 

# start and end image indices for training and validating 
startIndices_train = np.ones(n_cosmosToTrain).astype(int)
endIndices_train = n_trainimages*np.ones(n_cosmosToTrain).astype(int)
startIndices_validate = (n_trainimages+1)*np.ones(n_cosmosToTrain).astype(int)
endIndices_validate = (n_trainimages+n_validateimages)*np.ones(n_cosmosToTrain).astype(int)

learning_rate = 0.005


######################## define functions & Class ###############################

def read_parametersfromfile(path):
    '''
    Input: string for the directory to the folders of cosmology 
    
    Function: reads the Omega_m and sigma_8 values from the folder names in 
    the path directory and store them as arrays of strings. Stores the values 
    as strings to avoid rounding, for example, 0.600 to 0.6. 
    
    Output: (array of strings of Omega_m, array of strings of sigma_8)
    '''
    
    Om_strings = [] 
    si_strings = []
    
    for filename in os.listdir(path):
        if (filename.startswith('Om') and not filename.endswith('.gz')): 
            Om_strings.append(filename[2:7]) 
            si_strings.append(filename[10:15])
            
    return Om_strings, si_strings


def read_files(path, Oms, sis, start_indices, end_indices): 
    '''
    Input: file root directory path, strings of omegam and sigma8 values, 
    start image indices and end image indices. Oms, sis, start_indices, end_indices
    should be arrays of same length. 
    
    Function: reads the images labeled between start_indices[i] and end_indices[i] (inclusive)
    for each of the ith cosmologies specified by omegam and sigma8 values 
    
    Output: numpy array of 3D images
    '''
    
    n_cosmos = len(Oms)  # this should be shared by Oms, sis, start_indices, end_indices 
    
    # read in the files and put them into 3D matrix 
    filedata = []

    for j in range(0, n_cosmos): 
        i_start = start_indices[j]
        i_end = end_indices[j]
        Om = Oms[j]
        si = sis[j]
        
        for i in range(i_start, i_end+1): 
            if (i < 10): filenum = '00{}'.format(i)
            elif (i < 100): filenum = '0{}'.format(i)
            else: filenum = i 

            hdulist = fits.open('{}Om{}_si{}/WLconv_z1.00_0{}r.fits'.format(path, Om, si, filenum))
            image = hdulist[0].data

            # downsize from dim_image to dim_downsized 
            image_downsized = image.reshape([dim_downsized, dim_image//dim_downsized, 
                                    dim_downsized, dim_image//dim_downsized]).mean(3).mean(1)

            # newaxis makes the image 3D (1x512x512) 
            filedata.append(image_downsized[np.newaxis,:,:])
    
    return np.array(filedata)
           
    
def test_cnn(network, testinputs, batchsize, isTrackingloss, **kwargs): 
    '''
    Input: CNN object, testing data, batch size for testing, boolean for whether 
    to track losses bewteen test ouputs and targets, **kwargs includes the loss function
    criterion and the targetting outputs. Target should be in the form of 
    an array of (omegam, sigma8) parameters each corresponding to the target of one batch. 
    
    Call with 
    test_cnn(network, testinputs, batchsize, False) 
    or 
    test_cnn(network, testinputs, batchsize, True, loss_fn=loss_fn, target=target)
    
    Function: Run the network with test inputs and specified batch size and track loss as 
    requested. 
    
    Output: 
    If isTrackingloss is True, return (np array of predicted outputs, np array of losses for each batch)
    If isTrackingloss is False, return np array of predicted outputs
    '''
    
    # unpack optional arguments **kwargs 
    if (isTrackingloss): 
        loss_fn = kwargs.get('loss_fn', None)
        testtargets = kwargs.get('target', None)
        losses = []
    
    # make the data into 4D tensor and put each batch into iterable 
    testloader = torch.utils.data.DataLoader(testinputs, batch_size=batchsize, shuffle=False)
    
    predictions = []  # stores outputs of network with test data inputs 
    
    with torch.no_grad(): 
        for i, testdata in enumerate(testloader, 0):
            testdata = testdata.to(device)  # GPU
            testoutputs = network(testdata)
            predictions.append(((testoutputs[0]).cpu()).numpy())  # put tensor back into cpn before converting to np 

            if (isTrackingloss): 
                testtarget = torch.tensor(testtargets[i]).repeat(batchsize, 1)  
                testtarget = testtarget.to(device)  # GPU
                losses.append((loss_fn(testoutputs, testtarget)).cpu())
    
    if (isTrackingloss): 
        return np.array(predictions), np.array(losses)
    
    return np.array(predictions)


# network structure 

def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU()
    )

class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # layers 1+2 
            conv_block(1, 32, kernel_size=3), 
            conv_block(32, 32, kernel_size=3), 
            nn.AvgPool2d(2, 2), 
            # layers 3+4
            conv_block(32, 64, kernel_size=3), 
            conv_block(64, 64, kernel_size=3), 
            nn.AvgPool2d(2, 2), 
            # layers 5-7 
            conv_block(64, 128, kernel_size=3),
            conv_block(128, 64, kernel_size=1),
            conv_block(64, 128, kernel_size=3),
            nn.AvgPool2d(2, 2),
            # layers 8-10 
            conv_block(128, 256, kernel_size=3),
            conv_block(256, 128, kernel_size=1),
            conv_block(128, 256, kernel_size=3),
            nn.AvgPool2d(2, 2),
            # layers 11-13 
            conv_block(256, 512, kernel_size=3),
            conv_block(512, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3),
            nn.AvgPool2d(2, 2),
            # layers 14-18 
            conv_block(512, 512, kernel_size=3),
            conv_block(512, 256, kernel_size=1),
            conv_block(256, 512, kernel_size=3),
            conv_block(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),             # no batch norm on last convolution layer 
            nn.AvgPool2d(6, 6)
        )
        
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        
        return x
    

############################# Code ##################################

net = Net()
net.to(device)  #GPU

# loss function and training rate defined by paper 
criterion = nn.L1Loss()  # MAE loss 
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[10,15,20,25], gamma=0.1)     # set training schedule 

# read in all the omegam and sigma8 values from the directory 
Omegam_strings, sigma8_strings = read_parametersfromfile(rootpath)

Omegastrings_train = Omegam_strings[:n_cosmosToTrain]
sigmastrings_train = sigma8_strings[:n_cosmosToTrain]
Omegas_train = np.array(Omegastrings_train).astype(np.float)
sigmas_train = np.array(sigmastrings_train).astype(np.float)

# use the same sets of params for validation as training for now (can change as appropriate)
Omegastrings_validate = Omegastrings_train
sigmastrings_validate = sigmastrings_train
Omegas_validate = np.array(Omegastrings_validate).astype(np.float)
sigmas_validate = np.array(sigmastrings_validate).astype(np.float)


start_time = time.time()  
inputs_train = read_files(rootpath, Omegastrings_train, sigmastrings_train, 
                          startIndices_train, endIndices_train)
inputs_validate = read_files(rootpath, Omegastrings_validate, sigmastrings_validate, 
                             startIndices_validate, endIndices_validate)
print("--- Time to load the files: %s seconds ---" % (time.time() - start_time), flush=True)

# make the data into 4D tensor and put each batch into iterable 
trainloader = torch.utils.data.DataLoader(inputs_train, batch_size=n_perbatch, shuffle=False)


# train the network 
start_time = time.time()  # to time the training

losseslist_train = []
losseslist_validate = []

for epoch in range(n_epoch): 
    running_loss = 0.0   # running loss of training per epoch
    
    for i, data in enumerate(trainloader, 0): 
        data = data.to(device)  # GPU
        # update target value of sigma8 and omegam (stacked to the correct dimension to match the number per batch)
        if (i % n_batch_train == 0): 
            params_index = int(i//n_batch_train) 
            
            # change target every n_batch_train batches  
            target = torch.tensor([Omegas_train[params_index], sigmas_train[params_index]]).repeat(n_perbatch, 1)  
            target = target.to(device)  # GPU
        
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print('Epoch %d training loss: %.3f' % (epoch + 1, running_loss/len(trainloader)), flush=True)
    losseslist_train.append(running_loss/len(trainloader))

    # Validation 
    n_batch_validate = int(n_validateimages/n_perbatch_validate)
    targets_temp = np.concatenate((np.array([Omegas_validate]).T, np.array([sigmas_validate]).T), axis=1) # temp stacked targets of all comsos being trained
    targets_validate = np.repeat(targets_temp, n_batch_validate, axis=0)  # repeat each set of parameters by the number of batches
    outputs_validate, losses_validate = test_cnn(net, inputs_validate, n_perbatch_validate, 
                                                 True, loss_fn=criterion, target=targets_validate)
    
    print('Epoch %d validation loss: %.3f' % (epoch + 1, np.sum(losses_validate)/len(losses_validate)), flush=True)
    losseslist_validate.append(np.sum(losses_validate)/len(losses_validate))
    
    scheduler.step()  # update learning rate 


print("--- Training time: %s seconds ---" % (time.time() - start_time), flush=True)


# save the network 
PATH_save = '/n/holyscratch01/dvorkin_lab/Users/tianliwang/fullNet_gpu_90Cosmo_30epoch.pth'
torch.save(net.state_dict(), PATH_save) 

# save the losses 
np.save('/n/holyscratch01/dvorkin_lab/Users/tianliwang/losses_output.npy', [losseslist_train, losseslist_validate])

