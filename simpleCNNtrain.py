from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import time
import os

######################## define constants etc. ############################### 

rootpath = '/n/holyscratch01/dvorkin_lab/Users/tianliwang/maps_unzipped/'  # location of maps 

# these parameters are defined in the paper 
n_epoch = 10     # number of epochs to run 
n_imagespercosmo = 512  # number of images per cosmology 
n_perbatch = 32         # batch size
n_batch = n_imagespercosmo/n_perbatch  # number of minibatches 
dim_image = 1024 
dim_downsized = int(dim_image/2)            # downsized image dimension 

trainFraction = 0.6875  # fraction of images per cosmology for training 
testFraction = 1-trainFraction 
n_cosmosToTrain = 1  # number of cosmologies to take images from 
n_trainimages = int(n_imagespercosmo*trainFraction)   # number of images per cosmology to train

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
           
    
# network structure 

class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.pool6 = nn.AvgPool2d(6, 6)
        
        self.conv1to32_5 = nn.Conv2d(1, 32, 5)
        self.conv32to64_5 = nn.Conv2d(32, 64, 5)
        self.conv64to128_5 = nn.Conv2d(64, 128, 5)
        self.conv128to256_5 = nn.Conv2d(128, 256, 5)
        self.conv256to512_5 = nn.Conv2d(256, 512, 5)
        self.conv512to256_5 = nn.Conv2d(512, 216, 5)
        self.conv256to512_3 = nn.Conv2d(216, 512, 3)
        
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool2(F.relu(self.conv1to32_5(x)))  # out: 254x254x32 
        x = self.pool2(F.relu(self.conv32to64_5(x)))  # out: 125x125x64 
        x = self.pool2(F.relu(self.conv64to128_5(x)))  # out: 60x60x128 
        x = self.pool2(F.relu(self.conv128to256_5(x)))  # out: 28x28x256 
        x = self.pool2(F.relu(self.conv256to512_5(x)))  # out: 12x12x512
        x = F.relu(self.conv512to256_5(x))  # out: 8x8x256 
        x = self.pool6(F.relu(self.conv256to512_3(x)))  # out: 6x6x512->1x1x512
        
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    

############################# Code ##################################

net = Net()

# loss function and training rate defined by paper 
criterion = nn.L1Loss()  # MAE loss 
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# read in all the omegam and sigma8 values from the directory 
Omegam_strings, sigma8_strings = read_parametersfromfile(rootpath)

startIndices_train = np.ones(n_cosmosToTrain).astype(int)
endIndices_train = n_trainimages*np.ones(n_cosmosToTrain).astype(int)
Omegastrings_train = Omegam_strings[:n_cosmosToTrain]
sigmastrings_train = sigma8_strings[:n_cosmosToTrain]
Omegas_train = np.array(Omegastrings_train).astype(np.float)
sigmas_train = np.array(sigmastrings_train).astype(np.float)

start_time = time.time()  
inputs_train = read_files(rootpath, Omegastrings_train, sigmastrings_train, startIndices_train, endIndices_train)
print("--- Time to load the files: %s seconds ---" % (time.time() - start_time))

# make the data into 4D tensor and put each batch into iterable 
trainloader = torch.utils.data.DataLoader(inputs_train, batch_size=n_perbatch, shuffle=False)


# train the network 
start_time = time.time()  # to time the training

for epoch in range(n_epoch): 
    running_loss = 0.0 
    
    for i, data in enumerate(trainloader, 0): 
        # update target value of sigma8 and omegam (stacked to the correct dimension to match the number per batch)
        if (i % n_trainimages == 0): 
            params_index = i//n_trainimages  
            
            # change target every n_trainimages images 
            target = torch.tensor([Omegas_train[params_index], sigmas_train[params_index]]).repeat(n_perbatch, 1)  
        
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss/len(trainloader)))
        
print("--- Training time: %s seconds ---" % (time.time() - start_time))


# save the network 
PATH_save = '/n/holyscratch01/dvorkin_lab/Users/tianliwang/simpleNet_test_1cosmo.pth'
torch.save(net.state_dict(), PATH_save) 
