import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
import openslide
from Estimate_W import BLtrans
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from Estimate_W import Wfast

def run_stainsep(filename,nstains,lamb,output_direc="",background_correction=True):
	
	print 
	print "Running stain separation on:",filename

	level=0

	I = openslide.open_slide(filename)
	xdim,ydim=I.level_dimensions[level]
	img=np.asarray(I.read_region((0,0),level,(xdim,ydim)))[:,:,:3]

	print "Fast stain separation is running...."
	Wi,Hi,Hiv,stains=Faststainsep(I,img,nstains,lamb,level,background_correction)

	#print "\t \t \t \t \t \t Time taken:",elapsed

	print "Color Basis Matrix:\n",Wi

	fname=os.path.splitext(os.path.basename(filename))[0]
	cv2.imwrite(output_direc+fname+"-0_original.png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	cv2.imwrite(output_direc+fname+"-1_Hstain.png",cv2.cvtColor(stains[0], cv2.COLOR_RGB2BGR))
	cv2.imwrite(output_direc+fname+"-2_Estain.png",cv2.cvtColor(stains[1], cv2.COLOR_RGB2BGR))


def Faststainsep(I_obj,I,nstains,lamb,level,background_correction):
	s=I.shape
	ndimsI = len(s)
	if ndimsI!=3:
		print "Input Image I should be 3-dimensional!"
		sys.exit(0)
	rows = s[0]
	cols = s[1]

	num_patches=20
	patchsize=100

	#Estimate stain color bases + acceleration
	Wi,i0=Wfast(I_obj,nstains,lamb,num_patches,patchsize,level)


	if background_correction:
		print "Background intensity:",i0
	else:
		i0 = np.array([255.,255.,255.])
		print "Background correction disabled, default background intensity assumed"

	#Beer-Lambert tranformation
	V,VforW=BLtrans(I,i0)    #V=WH see in paper      
	Hiv=np.transpose(np.dot(np.linalg.pinv(Wi),np.transpose(V)))  #Pseudo-inverse
	Hiv[Hiv<0]=0

	Hi=np.reshape(Hiv,(rows,cols,nstains))
	#calculate the color image for each stain
	sepstains = []
	for i in range(nstains):
		vdAS =  np.reshape(Hiv[:,i],(rows*cols,1))*np.reshape(Wi[:,i],(1,3))
		sepstains.append(np.uint8(i0*np.reshape(np.exp(-vdAS), (rows, cols, 3))))
	return Wi,Hi,Hiv,sepstains
