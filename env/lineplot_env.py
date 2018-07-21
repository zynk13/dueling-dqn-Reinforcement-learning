import sys
sys.path.append("/home/aravind/Documents/Lineplot/automated_line_plot/script")
sys.path.append("/home/aravind/Documents/Lineplot/automated_line_plot/script/image-prop")
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import thread
import cv2
from scipy.stats import multivariate_normal
from scipy.stats import norm
from keras.applications.imagenet_utils import preprocess_input
from imageprop_utils import imageprop
import tf
image_utils=imageprop()
logger = logging.getLogger(__name__)
class lineplotEnv(gym.Env):
	metadata = {									###Not rendering the data, therefore not used here 5
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 30
	}

	def __init__(self):
		
		self.observation_space = spaces.Box(0,255,[30,30,3]) ## Image of size 224-224-3 
		self.action_space = spaces.Discrete(5) ### we define 8 possible actions 1) North 2)south 3)east 4)west 5)Northeast 6)northwest 7)Southeast 8)Southwest 
		self._seed()
		self.viewer = None
		self.state = None
		self.history_num=4    ###Define the length of history wanted 
		self.steps_beyond_done = None
		print("Initialization done for drone-v0")
		
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action): 
		#print("action predicted",action) 
		obs,done,out= image_utils.move(action)
		Img=np.asarray(obs,dtype=np.float64)
		#Img=np.resize(Img,(224,224,3))
		#Img = np.expand_dims(Img, axis=0)
		#Img=preprocess_input(Img)
		#print("Input size:",np.shape(Img))
		self.state=Img
		done=bool(done)
		if not done:
			reward = out
			print("reward given by the env:{}".format(reward))
		else:
			if self.steps_beyond_done == 0:
				logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further")
			#self.steps_beyond_done += 1
			print("2nd loop")
			reward = 0.0

		return np.array(self.state), reward, done, {}

#-----------------------------------------Reset----------------------------------------#
	def _reset(self):
		print("$$$$reset")
		
		im=image_utils.reset()
		#self.state = self.np_random.uniform(low=0, high=255, size=[240,320,3])
		'''
		stck=im
		for i in range(0,3):
			stck=np.dstack((stck,im))
		###Covert to the required image size###
		stck_pp = np.resize(stck_pp,(224,224,12))
		stck_pp=preprocess_input(stck_pp)
		print("input shape before transpose:{}".format(stck_pp.shape))
		#stck_pp=np.transpose(stck_pp)
		print("after transpose :{}".format(stck_pp))
		#self.state =stck_pp
		'''
		cv2.imshow("reset",im)
		cv2.waitKey(1)
		#im1=np.resize(im,(224,224,3))
		#print("im1:",im.shape)
		#im = np.expand_dims(im, axis=0)
		#print("data type of array:",np.dtype(im))
		#im1=preprocess_input(im)
		#print("input size:",np.shape(im1))
		
		self.state=im
		self.steps_beyond_done = None
		return np.array(self.state)

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * 1.0
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

		if self.state is None: return None

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
