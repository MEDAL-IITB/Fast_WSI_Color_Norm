import numpy as np
import time
import os
import cv2
import openslide

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from Estimate_W import Wfast


def run_batch_colornorm(filenames,nstains,lamb,output_direc,img_level,background_correction=True,config=None):	

	if config is None:
		config = tf.ConfigProto(log_device_placement=False)

	g_1 = tf.Graph()
	with g_1.as_default():
		Wis1 = tf.placeholder(tf.float32)
		Img1 = tf.placeholder(tf.float32,shape=(None,None,3))
		src_i_0 = tf.placeholder(tf.float32)
		
		s = tf.shape(Img1)
		Img_vecd = tf.reshape(tf.minimum(Img1,src_i_0),[s[0]*s[1],s[2]])
		V = tf.log(src_i_0+1.0) - tf.log(Img_vecd+1.0)
		Wi_inv = tf.transpose(tf.py_func(np.linalg.pinv, [Wis1], tf.float32))
		Hiv1 = tf.nn.relu(tf.matmul(V,Wi_inv))

		Wit1 = tf.placeholder(tf.float32)
		Hiv2 = tf.placeholder(tf.float32)
		sav_name = tf.placeholder(tf.string)
		tar_i_0 = tf.placeholder(tf.float32)
		normfac = tf.placeholder(tf.float32)
		shape = tf.placeholder(tf.int32)
		
		Hsonorm=Hiv2*normfac
		source_norm = tf.cast(tar_i_0*tf.exp((-1)*tf.reshape(tf.matmul(Hsonorm,Wit1),shape)),tf.uint8)
		enc = tf.image.encode_png(source_norm)
		fwrite = tf.write_file(sav_name,enc)

	session1=tf.Session(graph=g_1,config=config)


	file_no=0
	print "To be normalized:",filenames[1:],"using",filenames[0]
	for filename in filenames:

		display_separator()

		if background_correction:
			correc="back-correc"
		else:
			correc="no-back-correc"

		base_t=os.path.basename(filenames[0]) #target.svs
		fname_t=os.path.splitext(base_t)[0]   #target
		base_s=os.path.basename(filename)     #source.svs
		fname_s=os.path.splitext(base_s)[0]	  #source
		f_form = os.path.splitext(base_s)[1]  #.svs
		s=output_direc+base_s.replace(".", "_")+" (using "+base_t.replace(".", "_")+" "+correc+").png"
		# s=output_direc+base_s.replace(".", "_")+" (no-norm using "+base_t.replace(".", "_")+").png"
		#s=output_direc+fname_s+"_normalized.png"


		tic=time.time()
		print

		I = openslide.open_slide(filename)
		if img_level>=I.level_count:
			print "Level",img_level,"unavailable for image, proceeding with level 0"
			level=0
		else:
			level=img_level
		xdim,ydim=I.level_dimensions[level]
		ds=I.level_downsamples[level]

		if file_no==0:
			print "Target Stain Separation in progress:",filename,str(xdim)+str("x")+str(ydim)
		else:
			print "Source Stain Separation in progress:",filename,str(xdim)+str("x")+str(ydim)
		print "\t \t \t \t \t \t \t \t \t \t Time: 0"


		#parameters for W estimation
		num_patches=20
		patchsize=1000 #length of side of square 

		i0_default=np.array([255.,255.,255.],dtype=np.float32)

		Wi,i0 = Wfast(I,nstains,lamb,num_patches,patchsize,level,background_correction)
		if i0 is None:
			print "No white background detected"
			i0=i0_default

		if not background_correction:
			print "Background correction disabled, default background intensity assumed"
			i0=i0_default

		if Wi is None:
			print "Color Basis Matrix Estimation failed...image normalization skipped"
			continue
		print "W estimated",
		print "\t \t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
		Wi=Wi.astype(np.float32)

		if file_no==0:
			print "Target Color Basis Matrix:"
			print Wi
			Wi_target=np.transpose(Wi)
			tar_i0=i0
			print "Target Image Background white intensity:",i0
		else:
			print "Source Color Basis Matrix:"
			print Wi
			print "Source Image Background white intensity:",i0

		_max=2000
		
		print
		if (xdim*ydim)<=(_max*_max):
			print "Small image processing..."
			img=np.asarray(I.read_region((0,0),level,(xdim,ydim)),dtype=np.float32)[:,:,:3]
			
			Hiv=session1.run(Hiv1,feed_dict={Img1:img, Wis1: Wi,src_i_0:i0})
			# Hta_Rmax = np.percentile(Hiv,q=99.,axis=0)
			H_Rmax = np.ones((nstains,),dtype=np.float32)
			for i in range(nstains):
				t = Hiv[:,i]
				H_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)

			if file_no==0:
				file_no+=1
				Hta_Rmax = np.copy(H_Rmax)
				print "Target H calculated",
				print "\t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3)
				display_separator()
				continue

			print "Color Normalization in progress..."
			
			norm_fac = np.divide(Hta_Rmax,H_Rmax).astype(np.float32)
			session1.run(fwrite,feed_dict={shape:np.array(img.shape),Wit1: Wi_target,Hiv2:Hiv,sav_name:s,tar_i_0:tar_i0,normfac:norm_fac})

			print "File written to:",s
			print "\t \t \t \t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3)
			display_separator()

		else:
			_maxtf=3000
			x_max=xdim
			y_max=min(max(int(_maxtf*_maxtf/x_max),1),ydim)
			print "Large image processing..."
			if file_no==0:
				Hivt=np.memmap('H_target', dtype='float32', mode='w+', shape=(xdim*ydim,2))
			else:
				Hivs=np.memmap('H_source', dtype='float32', mode='w+', shape=(xdim*ydim,2))
				sourcenorm=np.memmap('wsi', dtype='uint8', mode='w+', shape=(ydim,xdim,3))
			x_tl = range(0,xdim,x_max)
			y_tl = range(0,ydim,y_max)
			print "WSI divided into",str(len(x_tl))+"x"+str(len(y_tl))
			count=0
			print "Patch-wise H calculation in progress..."
			ind=0
			perc=[]
			for x in x_tl:
				for y in y_tl:
					count+=1
					xx=min(x_max,xdim-x)
					yy=min(y_max,ydim-y)
					print "Processing:",count,"		patch size",str(xx)+"x"+str(yy),
					print "\t \t Time since processing started:",round(time.time()-tic,3)
					img=np.asarray(I.read_region((int(ds*x),int(ds*y)),level,(xx,yy)),dtype=np.float32)[:,:,:3]		

					Hiv = session1.run(Hiv1,feed_dict={Img1:img, Wis1: Wi,src_i_0:i0})
					if file_no==0:
						Hivt[ind:ind+len(Hiv),:]=Hiv
						_Hta_Rmax = np.ones((nstains,),dtype=np.float32)
						for i in range(nstains):
							t = Hiv[:,i]
							_Hta_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)
						perc.append([_Hta_Rmax[0],_Hta_Rmax[1]])
						ind+=len(Hiv)
						continue
					else:
						Hivs[ind:ind+len(Hiv),:]=Hiv
						_Hso_Rmax = np.ones((nstains,),dtype=np.float32)
						for i in range(nstains):
							t = Hiv[:,i]
							_Hso_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)
						perc.append([_Hso_Rmax[0],_Hso_Rmax[1]])
						ind+=len(Hiv)

			if file_no==0:
				print "Target H calculated",
				Hta_Rmax = np.percentile(np.array(perc),50,axis=0)
				file_no+=1
				del Hivt
				print "\t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
				ind=0
				continue

			print "Source H calculated",
			print "\t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
			Hso_Rmax = np.percentile(np.array(perc),50,axis=0)
			print "H Percentile calculated", 
			print "\t \t \t \t Time since processing started:",round(time.time()-tic,3)

			_normfac=np.divide(Hta_Rmax,Hso_Rmax).astype(np.float32)

			print "Color Normalization in progress..."
			count=0
			ind=0
			np_max=1000

			x_max=xdim
			y_max=min(max(int(np_max*np_max/x_max),1),ydim)
			x_tl = range(0,xdim,x_max)
			y_tl = range(0,ydim,y_max)
			print "Patch-wise color normalization in progress..."
			total=len(x_tl)*len(y_tl)
			
			prev_progress=0
			for x in x_tl:
				for y in y_tl:
					count+=1
					xx=min(x_max,xdim-x)
					yy=min(y_max,ydim-y)
					pix=xx*yy
					sh=np.array([yy,xx,3])
					
					#Back projection into spatial intensity space (Inverse Beer-Lambert space)
					
					sourcenorm[y:y+yy,x:x+xx,:3]=session1.run(source_norm,feed_dict={Hiv2: np.array(Hivs[ind:ind+pix,:]),Wit1:Wi_target,normfac:_normfac,shape:sh,tar_i_0:tar_i0})

					ind+=pix
					percent=5*int(count*20/total) #nearest 5 percent
					if percent>prev_progress and percent<100:
						print str(percent)+" percent complete...",
						print "\t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
						prev_progress=percent
			print "Color Normalization complete!",
			print "\t \t \t \t Time since processing started:",round(time.time()-tic,3)

			p = time.time()-tic
			s=output_direc+base_s.replace(".", "_")+" (using "+base_t.replace(".", "_")+" "+correc+").png"
			print "Saving normalized image..."
			cv2.imwrite(s,cv2.cvtColor(sourcenorm, cv2.COLOR_RGB2BGR))
			del sourcenorm
			print "File written to:",s
			print "\t \t \t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3)
			display_separator()

		file_no+=1
		if os.path.exists("H_target"):
			os.remove("H_target")
		if os.path.exists("H_source"):
			os.remove("H_source")
		if os.path.exists("wsi"):
			os.remove("wsi")

	session1.close()



def run_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level,background_correction=False,config=None):	
	filenames=[target_filename,source_filename]
	run_batch_colornorm(filenames,nstains,lamb,output_direc,level,background_correction,config)



def display_separator():
	print "________________________________________________________________________________________________"
	print