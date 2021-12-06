# Perceptual Consistency in Video Segmentation 
# https://arxiv.org/abs/2110.12385
import numpy as np
import torch
import torch.nn as nn
import skimage
from skimage.io import imread
import torchvision.models as models
import cv2

def SPC( pFeature_A , pFeature_B , seg_A , seg_B ): #Segmentation Perceptual Consistency:
  # pFeature_A : perceptual feature map for image A, shape =[c,w*h]
  # pFeature_A : perceptual feature map for image B, shape =[c,w*h]
  # seg_A : one - hot segmentation map for image A, shape =[ n_cls ,w*h]
  # seg_B : one - hot segmentation map for image B, shape =[ n_cls ,w*h]
  # normalize features in A to unit length :
  pFeature_A = nn.functional.normalize(pFeature_A , p =2 , dim =0)
  # normalize features in B to unit length :
  pFeature_B = nn.functional.normalize(pFeature_B , p =2 , dim =0)
  # prepare correct tensor shapes for computing correlation matrix :
  pFeature_A = pFeature_A.transpose_(1 ,0)
  seg_A = seg_A.transpose_(1 ,0)
  # compute correlation between perceptual features of two images
  correlation = torch.matmul (pFeature_A,pFeature_B)
  # find optimal matching of perceptual features w/o segmentation constraint (Eq .1 in the paper ):
  max0_no_constraint = torch.max(correlation, dim =1)
  max1_no_constraint = torch.max(correlation, dim =0)
  # find optimal matching of perceptual features under segmentation constraint (Eq .2 in the paper ):
  correlationSeg = torch.matmul( seg_A . float () , seg_B . float () )
  correlationSeg = correlation * correlationSeg
  max0_with_constraint = torch.max( correlationSeg , dim =1)
  max1_with_constraint = torch.max( correlationSeg , dim =0)

  # compute the averages to be used in Eq .3:
  mm0_avg = torch.mean ( correlation , dim =1)
  mm1_avg = torch.mean ( correlation , dim =0)
  
  # compute perceptual consistency (Eq .3 in the paper ):
  pcA_map =( max0_with_constraint [0] - mm0_avg ) /( max0_no_constraint [0] - mm0_avg )
  pcB_map =( max1_with_constraint [0] - mm1_avg ) /( max1_no_constraint [0] - mm1_avg )
  pcA_imageLevel = pcA_map . mean ()
  pcB_imageLevel = pcB_map . mean ()
  pc_overall = min ( pcA_imageLevel , pcB_imageLevel )
  #pdb.set_trace()
  return pc_overall, pcA_map , pcB_map
  

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#load ImageNet pretrained ResNet for feature extraction:
resnet18= models.resnet18(pretrained=True)
featureExtractor= torch.nn.Sequential(*list(resnet18.children())[:-4])

#a toy example:
id_A='00008'
id_B='00012'
#load two images:
img_A=imread('soapbox_imgs/'+id_A+'.jpg')
img_B=imread('soapbox_imgs/'+id_B+'.jpg')
img_A=skimage.img_as_float(img_A)
img_B=skimage.img_as_float(img_B)
img_A = np.transpose(img_A, (2,0,1))
img_B = np.transpose(img_B, (2,0,1))

img_A[0,:,:]=(img_A[0,:,:]-mean[0])/std[0]
img_A[1,:,:]=(img_A[1,:,:]-mean[1])/std[1]
img_A[2,:,:]=(img_A[2,:,:]-mean[2])/std[2]

img_B[0,:,:]=(img_B[0,:,:]-mean[0])/std[0]
img_B[1,:,:]=(img_B[1,:,:]-mean[1])/std[1]
img_B[2,:,:]=(img_B[2,:,:]-mean[2])/std[2]

img_A = torch.from_numpy(img_A).float()
img_B = torch.from_numpy(img_B).float()

img_A=torch.unsqueeze(img_A,0)
img_B=torch.unsqueeze(img_B,0)

#apply the feature extractor on these two images:
pFeature_A=featureExtractor(img_A)
pFeature_B=featureExtractor(img_B)

w=pFeature_A.shape[2]
h=pFeature_A.shape[3]

pFeature_A=torch.reshape(pFeature_A,(pFeature_A.shape[1],pFeature_A.shape[2]*pFeature_A.shape[3]))
pFeature_B=torch.reshape(pFeature_B,(pFeature_B.shape[1],pFeature_B.shape[2]*pFeature_B.shape[3]))

#load segmentation images (obtained from STM: https://github.com/seoungwugoh/STM)
seg_A_c1=cv2.resize(cv2.imread('soapbox_seg/' + id_A +'.png'),(w,h),interpolation=cv2.INTER_NEAREST)
seg_B_c1=cv2.resize(cv2.imread('soapbox_seg/'+ id_B + '.png'),(w,h),interpolation=cv2.INTER_NEAREST)
#construct the segmentation map:
seg_A_c1=seg_A_c1[:,:,2]
seg_B_c1=seg_B_c1[:,:,2]
seg_A_c1 = torch.from_numpy(seg_A_c1).float()
seg_B_c1 = torch.from_numpy(seg_B_c1).float()
seg_A_c1=torch.reshape(seg_A_c1,(1,seg_A_c1.shape[0]*seg_A_c1.shape[1]))
seg_B_c1=torch.reshape(seg_B_c1,(1,seg_B_c1.shape[0]*seg_B_c1.shape[1]))
seg_A_c1[seg_A_c1>0]=1
seg_B_c1[seg_B_c1>0]=1
seg_A_c2=1-seg_A_c1
seg_B_c2=1-seg_B_c1
seg_A=torch.cat((seg_A_c1,seg_A_c2),0)
seg_B=torch.cat((seg_B_c1,seg_B_c2),0)

pc_overall, pcA_map , pcB_map =SPC(pFeature_A , pFeature_B , seg_A , seg_B)
print (pc_overall)
