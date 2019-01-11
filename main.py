# Please cite both papers if you use this code: 
# @article{ramakrishnan_fast_2019,
# 	title = {Fast {GPU}-{Enabled} {Color} {Normalization} for {Digital} {Pathology}},
# 	url = {http://arxiv.org/abs/1901.03088},
# 	urldate = {2019-01-11},
# 	journal = {arXiv:1901.03088 [cs]},
# 	author = {Ramakrishnan, Goutham and Anand, Deepak and Sethi, Amit},
# 	month = jan,
# 	year = {2019},
# 	note = {arXiv: 1901.03088},
# 	keywords = {Computer Science - Computer Vision and Pattern Recognition},
# 	file = {arXiv\:1901.03088 PDF:/home/deepakanand/Zotero/storage/PBAERTSZ/Ramakrishnan et al. - 2019 - Fast GPU-Enabled Color Normalization for Digital P.pdf:application/pdf;arXiv.org Snapshot:/home/deepakanand/Zotero/storage/FH5SW7R3/1901.html:text/html}
# }


# @inproceedings{Vahadane2015ISBI,
# 	Author = {Abhishek Vahadane and Tingying Peng and Shadi Albarqouni and Maximilian Baust and Katja Steiger and Anna Melissa Schlitter and Amit Sethi and Irene Esposito and Nassir Navab},
# 	Booktitle = {IEEE International Symposium on Biomedical Imaging},
# 	Date-Modified = {2015-01-31 17:49:35 +0000},
# 	Title = {Structure-Preserved Color Normalization for Histological Images},
# 	Year = {2015}}

# input image should be color image
# Python implementation by: Goutham Ramakrishnan, goutham7r@gmail.com and Deepak Anand, deepakanandece@gmail.com

import glob
import sys
import os

from Run_StainSep import run_stainsep
from Run_ColorNorm import run_colornorm, run_batch_colornorm

#setting tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#to use cpu instead of gpu, uncomment the below line
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #use only GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '' #use only CPU

import tensorflow as tf
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
# config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)


#Parameters
nstains=2    #number of stains
lamb=0.01     #default value sparsity regularization parameter
# lamb=0 equivalent to NMF



# 1= stain separation of all images in a folder
# 0= stain separation of single image 
# 2= color normalization of one image with one target image
# 3= color normalization of all images in a folder with one target image
# 4= color normalization of one image with multiple target images
# 5= color normalization of all images in a folder with multiple target images individually
op=0

if op==0:
	filename="/home/deepakanand/Downloads/registered_data_for_color_normalization/1HE.tif"
	# filename="./experiment_CN/PrognosisTMABlock2_B_2_5_H&E1.tif"
	# filename="./Target/TCGA-E2-A14V-01Z-00-DX1.tif"
	# filename = "./Target/he.png"
	print filename
	run_stainsep(filename,nstains,lamb)


elif op==1:
	input_direc="./deepak_he/"
	output_direc="./stain_separated/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	file_type="*"
	if len(sorted(glob.glob(input_direc+file_type)))==0:
		print "No source files found"
		sys.exit()
	filenames=sorted(glob.glob(input_direc+file_type))
	print filenames
	for filename in filenames:
		run_stainsep(filename,nstains,lamb,output_direc=output_direc)


elif op==2:
	level=0
	output_direc="./new_tests/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)

	source_filename="./Test/WSI/57_HE: 10000x11534.png"
	# source_filename="./experiment_CN/PrognosisTMABlock3_A_5_6_H&E1.tif"
	#source_filename="./Test0/20_HE(28016,22316).png"
	#source_filename="./Source/TUPAC-TR-004.svs"
	#source_filename="../../Downloads/01/no1_HE.ndpi"
	#source_filename = "../../Documents/Shubham/Data/Test Data/78/78_HE.ndpi"
	target_filename="./Target/TCGA-E2-A14V-01Z-00-DX1.tif"
	# target_filename="b048.tif"

	if not os.path.exists(source_filename):
		print "Source file does not exist"
		sys.exit()
	if not os.path.exists(target_filename):
		print "Target file does not exist"
		sys.exit()
	background_correction = True	
	run_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level,background_correction,config=config)


elif op==3:
	level=0

	# input_direc="../experiment_CN/"
	input_direc="../Test/WSI patches/"
	output_direc="./new_tests_WSI/GPU/"
	if not os.path.exists(output_direc):
		os.makedirs(output_direc)
	file_type="*.*"
	#file_type="*.svs" #all of these file types from input_direc will be normalized
	target_filename="../Target/TCGA-E2-A14V-01Z-00-DX1.tif"
	if not os.path.exists(target_filename):
		print "Target file does not exist"
		sys.exit()
	if len(sorted(glob.glob(input_direc+file_type)))==0:
		print "No source files found"
		sys.exit()
	#filename format of normalized images can be changed in run_batch_colornorm
	filenames=[target_filename]+sorted(glob.glob(input_direc+file_type))
	
	background_correction = True
	# background_correction = False	
	run_batch_colornorm(filenames,nstains,lamb,output_direc,level,background_correction,config)
