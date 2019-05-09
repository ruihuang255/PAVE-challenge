import torch
import argparse
from torch.utils.data import DataLoader
from torch import optim
#from torchvision.transforms import transforms
from unet import Unet
from dataset import SSFPDataset, SSFPTestDataset
import nibabel as nib
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# if use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#parameter
parse=argparse.ArgumentParser()

def train_model(model, criterion, optimizer, dataload, num_epochs=120):
    
    parallel_model = nn.DataParallel(model, device_ids = [0,1])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        #print ( dt_size)
        #epoch_loss = 0
        step = 0
        for x, y, slice_num, pat_num, _ in dataload:
            step += 1        
            #print(x.size())
            inputs = x.to(device)
            #print(inputs.size())
            labels = y.to(device)
            #print(labels.size())
            label_1 = labels[:,0,:,:]
            label_2 = labels[:,1,:,:]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            #outputs = model(inputs)
            outputs = parallel_model(inputs)
            #print( outputs.shape)
            output_1 = outputs[:,0,:,:]
            output_2 = outputs[:,1,:,:]
            loss_1 = criterion(output_1, label_1)
            loss_2 = criterion(output_2, label_2)
            loss = loss_1 + loss_2
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            #epoch_loss += loss            
            
            #Accuracy           
            output_1[output_1>=0.5] = 1
            output_1[output_1<0.5] = 0           
            output_2[output_2>=0.5] = 1 
            output_2[output_2<0.5] = 0  
            #print(slice_num)
            #print(pat_num)
            #print(output_1.min())
            #print(output_1.max())
            #print(label_1.min())
            #print(label_1.max())
            #print(output_2.min())
            #print(output_2.max())
            #print(label_2.min())
            #print(label_2.max())
            dice_1 = float(torch.sum(output_1[label_1==1])*2.0) / float(output_1.sum() + label_1.sum())
            dice_2 = float(torch.sum(output_2[label_2==1])*2.0) / float(output_2.sum() + label_2.sum())
            #accuracy_1 = float((output_1==label_1).sum())/float(torch.numel(label_1))
            print("%d/%d,train_loss:%0.5f, Accuracy_vessel:%0.5f, Accuracy_artery:%0.5f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(),dice_1,dice_2))
           
            del outputs,inputs,labels,label_1,label_2

        #print("epoch %d loss:%0.10f" % (epoch, epoch_loss))
        #print("Loss is %0.6f" % (epoch_loss/num_epochs))
        #if epoch%10 == 9:
        #    torch.save(model.state_dict(), 'trained_model/e120_try2/weights_%d.pth' % epoch)
    return model

#train
def train():
    model = Unet(5, 2).to(device)
    model.train()
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    PAVE_dataset = SSFPDataset("train",transform=1,target_transform=1)
    dataloaders = DataLoader(PAVE_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders)

#test results
def save_nii(image, filename):
    nii_image = nib.Nifti1Image(image, None)
    nib.save(nii_image, filename)
    return

def test():
    model = Unet(5, 2)
    model.load_state_dict(torch.load(args.ckp,map_location='cpu'))
    model.to(device)
    
    test_root_dir = "test"
    PAVE_dataset = SSFPTestDataset(root=test_root_dir)
    # batch_size has to be divisible by 828 because there are 828 slices per patient
    batch_size = 1
    
    dataloaders = DataLoader(PAVE_dataset, batch_size=batch_size)
    model.eval()
    #import matplotlib.pyplot as plt
    #plt.ion()
    
    test_result_dir = "test_result"
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
        
    patients = np.zeros((1,832,2,224,832))
    
    with torch.no_grad():
        for x, slice_num, patient_num, leg in tqdm(dataloaders):
            x = x.to(device)
            y=model(x)
            output = y.cpu().numpy()  
            
            if leg[0] == 'left':
                patients[patient_num, slice_num+2,:,:192,80:400] = output[0,:,:,:]
            else: 
                patients[patient_num, slice_num+2,:,:192,480:800] = output[0,:,:,:]
    
    for patient_num in range(10):
        image_filename = os.path.join("/home/mng/scratch/PAVE_Challenge/test/", 'case{}'.format(patient_num+1), 'ssfp.nii.gz')

        size_x = nib.load(image_filename).shape[2]

        patient_output_vessels = np.transpose((patients[patient_num,:,0,:size_x,:] >= 0.5), axes=(2,0,1))
        patient_output_arteries = np.transpose((patients[patient_num,:,1,:size_x,:] >= 0.5), axes=(2,0,1))
        patient_output_veins = np.logical_and(patient_output_vessels, np.logical_not(patient_output_arteries))

        results_file = os.path.join(test_result_dir, 'case{}_results_vessels.nii.gz'.format(patient_num+1))
        save_nii(patient_output_vessels.astype(np.uint8), results_file)

        results_file = os.path.join(test_result_dir, 'case{}_results_arteries.nii.gz'.format(patient_num+1))
        save_nii(patient_output_arteries.astype(np.uint8), results_file)

        results_file = os.path.join(test_result_dir, 'case{}_results_veins.nii.gz'.format(patient_num+1))
        save_nii(patient_output_veins.astype(np.uint8), results_file)

    #plt.imshow(img_y)
    #plt.pause(0.01)
    #plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train()
    elif args.action=="test":
        test()
