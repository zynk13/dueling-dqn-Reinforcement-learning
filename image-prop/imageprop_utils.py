import numpy as np
import cv2
import os
import sys
import random
import matplotlib.pyplot as plt
import math

sys.path.append("/home/aravind/Documents/Lineplot/automated_line_plot/script")
sys.path.append("/home/aravind/Documents/Lineplot/automated_line_plot/script/dataset_generator")
sys.path.append("/home/aravind/Documents/Lineplot/automated_line_plot/script/image-prop/hough-line")

from hough_line import extract_axis
from extract_gtruth import Ground_Truth
#file='/home/nagashri/RL_drone/data/train/1/image.png'
#I=cv2.imread(file)
### This class of operations is only used to interact with the image
class imageprop():
	def __init__(self):
		print("init")
		self.current_gtruth=None
		self.current_img=None
		self.current_frame=None
		self.current_trace=None
		####Getting list of image names and shuffling them  to randomize data###
		self.inputpath="/home/aravind/Documents/Lineplot/automated_line_plot/data/train/dataset/images"
		path, dirs, files = os.walk(self.inputpath).next()
		self.fnames=[fname for fname in files if fname.endswith('.png')]
		self.arr=np.arange(len(self.fnames))
		np.random.shuffle(self.arr)
		groundtruth='/home/aravind/Documents/Lineplot/automated_line_plot/data/train/dataset/pixelgroundtruth.txt'
		self.g_truth = Ground_Truth(groundtruth)
		self.terminal=0
		self.current_frame=1
		self.current_trace_ID=0
		self.past_trace_ID=0
		self.pred_ID=1
		self.pred=[]
		self.past_trace=[]
		self.done=0
		self.pixelbias=[]
		self.pixeltrace=[]
		self.tracedone=0
	def reset(self):
		print("reset")
		filename=os.path.join(self.inputpath,self.fnames[self.arr[self.current_frame]]) #Get the next consecutive frame in the training data set
		print(filename)
		I=cv2.imread(filename)
		self.orig_img=I.shape
		I1,x,y=extract_axis(I)
		self.current_gtruth = self.g_truth.get_gtruth_of_image(self.fnames[self.arr[self.current_frame]])
		self.pixeltrace=[]
		self.current_img=I1
		self.pred=np.zeros((self.current_img.shape[0],self.current_img.shape[1]))
		self.past_trace=np.zeros((self.current_img.shape))
		self.pixelbias=[x,y]
		self.current_frame+=1
		self.getwindow()
		return self.current_trace
		#self.current_window=current_window
		#self.current_trace=current_trace



	def getwindow(self):
		##Scan with a pixel stride of 1 To do: param this
		self.current_window=[]
		print(self.current_img.shape)
		print("getwindow")
		self.done=1
		for x in range(30,self.current_img.shape[1]-30,10):				#To do: make this relative to the size of image pixels
			for y in range(30,self.current_img.shape[0]+30,10):
				window=[x,y,x+30,y-30] ### window size is 2 To do: Param this
				curr_img=self.current_img.copy()
				cv2.rectangle(curr_img,(x,y),(x+30,y-30),(0,0,255),2)
				cv2.imshow("image:",curr_img)
				cv2.waitKey(1)
				img_window=self.current_img[y-30:y,x:x+30].copy()
				gray = cv2.cvtColor(img_window,cv2.COLOR_BGR2GRAY)
				#edges = cv2.Canny(gray,50,150,apertureSize = 3)
				cv2.imshow("selected window:",gray)
				cv2.waitKey(1)
				if gray.min()<=180:   ## Contains any non white pixels
					#print("minimum:",gray.min())
					#ID=current_groundtruth(window)

					ID=self.get_labels_from_window(window)
					print("ID:",ID)
					if len(ID)>0:
						self.current_trace=img_window
						self.current_trace_ID=ID[0] 
						print("current trace ID:",ID[0])
						#unique=self.tracecheck(window,ID)  ####check if trace belongs to any past traces
						#if unique:
						self.current_window.append(window)
						self.done=0
						return
					#self.current_trace=img_window # To do write a code to get the reward from the system
					  #####To Do write code to get the current groundtruth from the window patch
						#return 
					#else:
						#print("trace already exists")
				#else:
					#print("no plots found, this plot is traced. Next image !!")
					

	
					
				
	def move(self,act):	### Function to move the window from the current position to the enxt defined by act
		#assert(self.current_window)
		#assert(self.current_img.any())
		#assert(self.current_trace.any())
		#print("act in move:",act)
		window=self.current_window[len(self.current_window)-1]

		if act==0: #move north
			x=window[0]
			y=window[1]-20
		elif act==1: #Move south
			x=window[0]
			y=window[1]+20
		
		elif act==2: #Move east
			x=window[0]+20
			y=window[1]

		elif act==3: #move Northeast
			x=window[0]+20
			y=window[1]-20


		elif act==4: #Move southeast
			x=window[0]+20
			y=window[1]+20

		else:
			x=window[0]
			y=window[1]
		####To DO : write code for reward######
		#print("image shape:",self.current_img.shape)
		#----------------Defining terminating stages -------------#
		#----checking conditions for image boundaries---------# 
		if x >= (self.current_img.shape[1]-30):
			x=self.current_img.shape[1]-30
			self.terminal+=1
			print("TERMINAL STAGE1")
		elif x <= 0:
			x=0
			self.terminal+=1
			print("TERMINAL STAGE2")
		if y >= self.current_img.shape[0]:
			y=self.current_img.shape[0]
			self.terminal+=1
			print("TERMINAL STAGE3")
		elif y <= 30:
			y=30
			self.terminal+=1
			print("TERMINAL STAGE4")
		print("stage1")

		window_new=[x,y,x+30,y-30]
		reward=self.getreward(window_new)
		#print("reward:",reward)
		self.current_window.append(window_new)
		img_window=self.current_img[y-30:y,x:x+30]
		####Condition for checking if there is no plot in the detected window 
		#print("current window:",self.current_window)
		gray = cv2.cvtColor(img_window,cv2.COLOR_BGR2GRAY)
		#cv2.imshow("selected window:",gray)
		#cv2.waitKey(1)
		curr_img=self.current_img.copy()
		#print("x:",x)
		#print ("y:",y)
		cv2.rectangle(curr_img,(x,y),(x+30,y-30),(0,0,255),2)
		cv2.imshow("image:",curr_img)
		cv2.waitKey(1)
		if  gray.min()<=180:   ## Contains any non white pixels
			self.tracedone=0
		else:	
			self.tracedone+=1
			reward=-1

		if self.tracedone==2 or self.terminal==2: # Two continuously empty windows then the plot is done 
			for x,y,x1,y1 in self.current_window:
				self.pred[y-20,x+20]=self.pred_ID	# ID of the completed trace 
				self.past_trace[y-30:y,x:x+30,:]=self.current_img[y-30:y,x:x+30,:].copy() ## Pixel values of the completed trace used to replace
				#self.current_img[x:x+20,y-30:y,:]=[255,255,255]	# Replace the pixels with the white value
				self.current_img[y-30:y,x:x+30,:]=[255,255,255]
			reward = -1
			curr_img=self.current_img.copy()
			#cv2.imshow("image after deletion:",curr_img)
			#cv2.waitKey(1)
			#print("size of self.pred:",self.pred.shape)
			i1,j1=np.where(self.pred==self.pred_ID)
			#plt.plot(i1,j1,'ro')
			#plt.show()
			cv2.imshow("Past_trace",self.past_trace)
			#cv2.waitKey(1)
			self.pred_ID+=1
			self.terminal=0
			self.getwindow() # Start searching for the new plot 

		return img_window,self.done,reward


	
	def getreward(self,window):
		#print("current trace ID:",self.current_trace_ID)
		#return 1 if self.current_trace_ID in self.get_labels_from_window(window) else 0

		g_truth_of_window = {pixel:labels for pixel,labels in self.current_gtruth.iteritems() 
					if pixel[1] >(window[0]+self.pixelbias[0]) and (pixel[0]) <((window[1])+self.pixelbias[1]) 
					and pixel[1] <(window[2]+self.pixelbias[0]) and (pixel[0]) >((window[3])+self.pixelbias[1])
					}

		pixels = [pixel for pixel,labels in g_truth_of_window.iteritems() if str(self.current_trace_ID) in labels]
		print pixels		
		if not pixels:
			return 0

		centroid_of_truth = [np.mean(x) for x in zip(*pixels)]
		center_of_window = [(window[3]+window[1]+2*self.pixelbias[1])/2,(window[2]+window[0]+2*self.pixelbias[0])/2]
		print 'center of window '+ str(center_of_window) +' centroid '+str(centroid_of_truth)
		dist = math.sqrt(math.pow(centroid_of_truth[0]-center_of_window[0],2) + math.pow(centroid_of_truth[1]-center_of_window[1],2))
		print 'dist = '+str(dist)
		return (1 - (dist/21.5))

	def get_labels_from_window(self,window):

		labels_in_window = set()
		for pixel,labels in self.current_gtruth.iteritems():
			#print(self.pixeltrace)
			#print("image dimension:",self.current_img.shape)
			#print("window:",window)
			#print("pixel bias:",self.pixelbias)
			#if pixel[1] >= (window[0]+self.pixelbias[0]) and pixel[0] >= ((window[1]+self.pixelbias[1])) and pixel[1] <= (window[2]+self.pixelbias[0]) and pixel[0] <= ((window[3]+self.pixelbias[1])):
			if pixel[1] >(window[0]+self.pixelbias[0]) and (pixel[0]) <((window[1])+self.pixelbias[1]) and pixel[1] <(window[2]+self.pixelbias[0]) and (pixel[0]) >((window[3])+self.pixelbias[1]):
				labels_in_window.update(map(int,labels.split(',')))
				#print("labels_in_window",labels_in_window)
		return list(labels_in_window)
