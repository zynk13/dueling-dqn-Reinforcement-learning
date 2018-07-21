from numpy import *
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import cycle

def linepixels(x0, y0, x1, y1,ht):

        x=[]
        y=[]

        steep = abs(x1-x0) < abs(y1-y0)

        if steep:
            x0,y0 = y0,x0
            x1,y1 = y1,x1

        if x0 > x1:
            x0,x1 = x1,x0
            y0,y1 = y1,y0
	
	dx = x1-x0
        dy = y1-y0

	if dx==0:
		gradient = 1.0
	else:
        	gradient = float(dy) / float(dx)  # slope

        """ handle first endpoint """
        xend = round(x0)
        yend = y0 + gradient * (xend - x0)
        xpxl0 = int(xend)
        ypxl0 = int(yend)
        x.append(xpxl0)
        y.append(ypxl0) 
        x.append(xpxl0)
        y.append(ypxl0+1)
        intery = yend + gradient

        """ handles the second point """
        xend = round (x1);
        yend = y1 + gradient * (xend - x1);
        xpxl1 = int(xend)
        ypxl1 = int(yend)
        x.append(xpxl1)
        y.append(ypxl1) 
        x.append(xpxl1)
        y.append(ypxl1 + 1)

        """ main loop """
        for px in range(xpxl0 + 1 , xpxl1):
            x.append(px)
            y.append(int(intery))
            x.append(px)
            y.append(int(intery) + 1)
            intery = intery + gradient;

        if steep:
            y,x = x,y
	
	#y = map(lambda z:ht-z, y)

        coords=zip(x,y)

        return coords


lines = ["-","--","-.",":"]
linecycler = cycle(lines)
t = linspace(0.05,2*math.pi,60)
output_file = open('pixelgroundtruth.txt', 'w')

#ax = plt.subplot(111)

for pltnum in range(5000):
	
	lbl = ''
	ctr = 0
	g_truth = {}
	coords = []
	image_name = 'sample'+str(pltnum)+'.png'

	fig, ax = plt.subplots()

	ax.set_xlim([-0.5,6.5])
	ax.set_ylim([-4,4])

	_, ht = fig.canvas.get_width_height()

	plt.xlabel('xval')
	plt.ylabel('yval')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	n = random.randint(2,4)
	
	option_list = [1,2,3,4,5]

	for i in range(n):
	
		ctr+=1
		coord = []

		noise = np.random.normal(0,random.uniform(0.05,0.25),60) + random.uniform(-1,1)

		#fn = input("Enter the respective num for function:\n1 - sin\n2 - cos\n3 - log\n4 - sin+cos\n5 - sin*cos\n")
	
		repeat_probability = random.random()

		if repeat_probability > 0.8:
			fn = random.randint(1,5)
			if fn in option_list:
				option_list.remove(fn)
		else:
			fn = random.choice(option_list)
			option_list.remove(fn)

		if fn is 1:
			a = sin(t) + noise
			lbl = str(ctr)
			#lbl = 'sin'+str(ctr)
		elif fn is 2:
			a = cos(t) + noise
			lbl = str(ctr)
			#lbl = 'cos'+str(ctr)
		elif fn is 3:
			a = log(t) + noise
			lbl = str(ctr)
			#lbl = 'log'+str(ctr)
		elif fn is 4:
			a = sin(t) + cos(t) + noise
			lbl = str(ctr)
			#lbl = 'sin+cos'+str(ctr)
		elif fn is 5:
			a = (sin(t) * cos(t)) + noise
			lbl = str(ctr)
			#lbl = 'sin*cos'+str(ctr)

		ax.plot(t, a, next(linecycler), label=lbl)

		#g_truth[lbl] = [(int(round(ele[0])),int(round(ht-ele[1]))) for ele in 
				#ax.transData.transform(zip(t,a))]
		#samp = [(int(round(ele[0])),int(round(ele[1]))) for ele in ax.transData.transform(zip(t,a))]

		xy_pixels = ax.transData.transform(np.vstack([t,a]).T)
		xpix, ypix = xy_pixels.T

		points = [(int(round(ele[0])),int(round(ele[1]))) for ele in zip(xpix,ypix)]

		flg = 0
		for ele in points:
			if flg==1:	
				coord += linepixels(prev[0],prev[1],ele[0],ele[1],ht)
			else:
				flg=1
			prev = ele

		coords.append(coord)

	#pixel_map = np.zeros((640,480),dtype=str)
	pixel_map = {}
	plotnum = 0
	for plot in coords:
		plotnum+=1
		for coord in plot:
			if coord not in pixel_map:
				pixel_map[coord] = str(plotnum)
			else:
				if str(plotnum) not in pixel_map[coord]:
					pixel_map[coord] = pixel_map[coord] + ',' + str(plotnum)
	"""print pixel_map

	plt.figure()
	plt.imshow(pixel_map,interpolation='none',
		    origin='lower',
		    cmap='gist_earth_r',
		    vmin=0,
		    vmax=1)
	plt.show()"""

	output_file.write(image_name+'-')
	for key,val in pixel_map.iteritems():
		output_file.write(str(key[0])+','+str(key[1])+':'+str(val)+' ')
	output_file.write('\n')

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.savefig('dataset/'+image_name)
	plt.close(fig)
