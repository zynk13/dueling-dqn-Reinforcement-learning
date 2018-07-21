import numpy as np 
import os
from imageprop_utils import imageprop
from keyboard_utils import _Getch
import cv2
obj=imageprop()
obj.reset()
inkey = _Getch()
act=0
img,check,reward=obj.move(act)
while(1):
	img=[]
	print("while")
	k=inkey()
	#if k!='':break
	if k=='\x1b[A':
		print "up"
		act=0
	elif k=='\x1b[B':
		print "down"
		act=1
	elif k=='\x1b[C':
		print "right"
		act=2
	else:
		print "not an arrow key!"
		exit()
	print("act:",act)
	img,check,reward=obj.move(act)
	print("reward",reward)
	cv2.imshow("current trace:",img)
	cv2.waitKey(0)

	
