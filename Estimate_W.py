import sys
import os
import spams
import numpy as np
import math
from sklearn import preprocessing
from multiprocessing import Pool
from functools import partial
import signal

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)



def Wfast(img,nstains,lamb,num_patches,patchsize,level,background_correction=False):
	
	param=definePar(nstains,lamb)
	_max=3000
	max_size=_max*_max
	xdim,ydim=img.level_dimensions[0]
	patchsize=int(min(patchsize,xdim/3,ydim/3))
	patchsize_original=patchsize
	nstains=param['K']
	valid_inp=[]
	
	white_pixels=[]

	#100,000 pixels or 20% of total pixels is maximum number of white pixels sampled
	max_num_white=min(100000,(xdim*ydim)/5)
	min_num_white=10000

	white_cutoff=220
	I_percentile=90

	if ydim*xdim>max_size:
		print "Finding patches for W estimation:"
		for j in range(20):
			#print "Patch Sampling Attempt:",i+1
			initBias=int(math.ceil(patchsize/2)+1) 
			xx=np.array(range(initBias,xdim-initBias,patchsize))
			yy=np.array(range(initBias,ydim-initBias,patchsize))
			xx_yy=np.transpose([np.tile(xx, len(yy)), np.repeat(yy, len(xx))])
			np.random.shuffle(xx_yy)

			threshold=0.1 #maximum percentage of white pixels in patch
			for i in range(len(xx_yy)):
				patch=np.asarray(img.read_region((xx_yy[i][0],xx_yy[i][1]),level,(patchsize,patchsize)))
				patch=patch[:,:,:3]
				if len(white_pixels)<max_num_white:
					white_pixels.extend(patch[np.sum((patch>white_cutoff),axis=2)==3])

				if patch_Valid(patch,threshold):
					valid_inp.append(patch)
					if len(valid_inp)==num_patches:
						break

			if len(valid_inp)==num_patches:
				white_pixels=np.array(white_pixels[:max_num_white])
				break																																																																																																																																	
			patchsize=int(patchsize*0.95)
		valid_inp=np.array(valid_inp)
		print "Number of patches sampled for W estimation:", len(valid_inp)
	else:
		patch=np.asarray(img.read_region((0,0),level,(xdim,ydim)))
		patch=patch[:,:,:3]
		valid_inp=[]
		valid_inp.append(patch)
		white_pixels= patch[np.sum((patch>white_cutoff),axis=2)==3]
		print "Image small enough...W estimation done using whole image"

	if background_correction:
		print "Number of white pixels sampled",len(white_pixels)
		if len(white_pixels)<min_num_white:
			i0=np.array([255.0,255.0,255.0])
			print "Not enough white pixels found, default background intensity assumed"
		elif len(white_pixels)>0:
			i0 = np.percentile(white_pixels,I_percentile,axis=0)[:3]
		else:
			i0 = None
	else:
		i0 = np.array([255.0,255.0,255.0])

	if len(valid_inp)>0:
		out = suppress_stdout()
		pool = Pool(initializer=initializer)
		try:
		    WS = pool.map(partial(getstainMat,param=param,i_0=i0),valid_inp)
		except KeyboardInterrupt:
			pool.terminate()
			pool.join()
		pool.terminate()
		pool.join()
		suppress_stdout(out)

		WS=np.array(WS)

		if WS.shape[0]==1:
			Wsource=WS[0,:3,:]
		else:
			print "Median color basis of",len(WS),"patches used as actual color basis"
			Wsource=np.zeros((3,nstains))
			for k in range(nstains):
			    Wsource[:,k]=[np.median(WS[:,0,k]),np.median(WS[:,1,k]),np.median(WS[:,2,k])]
		
		Wsource = W_sort(normalize_W(Wsource,nstains))

		if Wsource.sum()==0:
			if patchsize*0.95<100:
				print "No suitable patches found for learning W. Please relax constraints"
				return None			#to prevent infinite recursion
			else:
				print "W estimation failed, matrix of all zeros found. Trying again..."				
				return Wfast(img,nstains,lamb,min(100,num_patches*1.5),int(patchsize_original*0.95),level)
		else:
			return Wsource,i0
	else:
		print "No suitable patches found for learning W. Please relax constraints"
		return None,None

def patch_Valid(patch,threshold):
	r_th=220
	g_th=220
	b_th=220
	tempr = patch[:,:,0]>r_th
	tempg = patch[:,:,1]>g_th
	tempb = patch[:,:,2]>b_th
	temp = tempr*tempg*tempb
	r,c = np.shape((temp)) 
	prob= float(np.sum(temp))/float((r*c))
	#print prob
	if prob>threshold:
		return False
	else:
		return True  

def W_sort(W):
	# All sorting done such that first column is H, second column is E
	# print W

	method = 3

	if method==1:
		# 1. Using r values of the vectors. E must have a smaller value of r (as it is redder) than H
		W = W[:,np.flipud(W[0,:].argsort())]
	elif method==2:
		# 2. Using b values of the vectors. H must have a smaller value of b (as it is bluer) than E
		W = W[:,W[2,:].argsort()]
	elif method==3:
		# 3. Using r/b ratios of the vectors. H must have a larger value of r/b.
		r_b_1 = W[0][0]/W[2][0]
		r_b_2 = W[0][1]/W[2][1]
		# print r_b_1, r_b_2
		if r_b_1<r_b_2: #else no need to switch
			W[:,[0, 1]] = W[:,[1, 0]]
	elif method==4:
		# 4. Using r-b values of the vectors. H must have a larger value of r-b.
		# This is equivalent to comparing the ratios of e^(-r)/e^(-b)
		r_b_1 = W[0][0]-W[2][0]
		r_b_2 = W[0][1]-W[2][1]
		# print r_b_1, r_b_2
		if r_b_1<r_b_2: #else no need to switch
			Wsource[:,[0, 1]] = Wsource[:,[1, 0]]

	return W

def BLtrans(Ivecd,i_0):
	Ivecd = vectorise(Ivecd)
	V=np.log(i_0)- np.log(Ivecd+1.0)
	w_threshold=220
	c = (Ivecd[:,0]<w_threshold) * (Ivecd[:,1]<w_threshold) * (Ivecd[:,2]<w_threshold)
	Ivecd=Ivecd[c]
	VforW=np.log(i_0)- np.log(Ivecd+1.0) #V=WH, +1 is to avoid divide by zero
	#shape of V = no. of pixels x 3 
	return V,VforW

def getstainMat(I,param,i_0):
	#I : Patch for W estimation
	V,VforW=BLtrans(I,i_0)   #Beer-Lambert law
	#step 2: Sparse NMF factorization (Learning W; V=WH)
	out = suppress_stdout()
	Ws = spams.trainDL(np.asfortranarray(np.transpose(VforW)),**param)
	suppress_stdout(out)
	return Ws

def normalize_W(W,k):
	W1 = preprocessing.normalize(W, axis=0, norm='l2')
	return W1

def definePar(nstains,lamb,batch=None):

	param={}	
	#param['mode']=2               #solves for =min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2 + ... 
								   #lambda||alpha_i||_1 + lambda_2||alpha_i||_2^2
	param['lambda1']=lamb
	#param['lambda2']=0.05
	param['posAlpha']=True         #positive stains 
	param['posD']=True             #positive staining matrix
	param['modeD']=0               #{W in Real^{m x n}  s.t.  for all j,  ||d_j||_2^2 <= 1 }
	param['whiten']=False          #Do not whiten the data                      
	param['K']=nstains             #No. of stain = 2
	param['numThreads']=-1         #number of threads
	param['iter']=40               #20-50 is OK
	param['clean']=True
	if batch is not None:
		param['batchsize']=batch   #Give here input image no of pixels for traditional dictionary learning
	return param

def vectorise(I):
	s=I.shape
	if len(s)==2: #case for 2D array
		third_dim=1
	else:
		third_dim=s[2] 
	return np.reshape(I, (s[0]*s[1],third_dim))

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def suppress_stdout(out=None):
	if out is None:
		devnull = open('/dev/null', 'w')
		oldstdout_fno = os.dup(sys.stdout.fileno())
		os.dup2(devnull.fileno(), 1)
		return oldstdout_fno
	else:
		os.dup2(out, 1)
