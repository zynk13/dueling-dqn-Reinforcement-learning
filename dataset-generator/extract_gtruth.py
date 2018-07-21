
class Ground_Truth():

	def __init__(self,filename):
		# Creates a ground truth dictionary on first call and saves all ground truth values from file in this dictionary

		self.g_truth = {}
		with open(filename, 'r') as f:
			for line in f.readlines():
				image,imageval = line.split('-')
				self.g_truth[image] = {}
				for ele in imageval.split(' '):
					if ele != '\n':
						key,val = ele.split(':')
						self.g_truth[image][tuple(map(int, key.split(',')))] = val

	def is_plot(self,image,pixel,plotnum):
		# image - image name
		# pixel - tuple of the current pixel to check plotnum on
		# plotnum - string value of the plotnumber to check on
		
		return plotnum in self.g_truth[image][pixel]
	
	def get_gtruth_of_image(self,imagename):
		# returns the ground truth dictionary for a particular image
		#print(list(self.g_truth))
		#print(len(self.g_truth))

		pixel_map = {}

		for pixel,label in self.g_truth[imagename].iteritems():
			pixel_map[(480-pixel[1],pixel[0])] = label

		return pixel_map

# Direction for accessing the class and function:

""" g_truth = Ground_Truth('pixelgroundtruth.txt') """

# Create an object for the ground truth class by giving it the respective file. This reads the contents of the file and saves it as a dictionary

""" print g_truth.is_plot('sample0.png',tuple((122,271)),'1') """

# Call the is_plot function to check if a pixel has a line plot in it or not

"""
Code for ground truth extract function :

>>>>>>> 08e473f42fc0960fba990ce571458ff629436396
def extract_ground_truth(filename):
	g_truth = {}
	with open(filename, 'r') as f:
		for line in f.readlines():
<<<<<<< HEAD
			try:
				key,val = line.split(':')
				g_truth[key] = [tuple(map(int, tup.split(','))) for tup in val.split()]
			except:
				print("not data")
	return g_truth

#print(extract_ground_truth('pixelgroundtruth.txt'))
=======
			image,imageval = line.split('-')
			g_truth[image] = {}
			for ele in imageval.split(' '):
				if ele != '\n':
					key,val = ele.split(':')
					g_truth[image][tuple(map(int, key.split(',')))] = val
	print g_truth.keys()
	return g_truth

extract_ground_truth('pixelgroundtruth.txt')"""

