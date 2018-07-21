import numpy as np
import gym
import gym_lineplot
import tensorflow as tf

from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.models import Model,load_model
from keras.layers import Input, merge, ZeroPadding2D,Concatenate,PReLU,Lambda
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
# from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D,Conv3D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D,MaxPooling3D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.utils import np_utils
from keras.engine.topology import Layer

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class L2Norm(Layer):

    def __init__(self, alpha=None, **kwargs):
        # self.output_dim = output_dim
        self.alpha = alpha
        super(L2Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.alpha is None:
            self.alpha_train = self.add_weight(name='alpha_train',
                                          shape=(1, 1),
                                          initializer='uniform',
                                          trainable=True)
        super(L2Norm, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x = K.transpose(x)
        unit_norm = x / (K.sqrt(K.sum(K.square(x), axis=0)) + 1e-7)
        # unit_norm = x / ((K.sum(K.abs(x), axis=0)) + 1e-7)
        # unit_norm = x / ((K.max(K.abs(x), axis=0)) + 1e-7)
        unit_norm = K.transpose(unit_norm)

        a = self.alpha
        if a is None:
            a = self.alpha_train
        
        return unit_norm * a
        
        
    def get_config(self):
        config = {
            'alpha': self.alpha
        }
        base_config = super(L2Norm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def residual_block(x,nb_filter,kernel_size,b,i):

    y = Conv3D(nb_filter,kernel_size, padding='same', name='block%d_conv%d'%(b,i), trainable=True)(x)
    # x = BatchNormalization(name='block1_conv1_bn')(x)
    y = Activation('relu', name='block%d_conv%d_act'%(b,i))(y)
    y = Concatenate(axis = -1)([x,y])
    print("y shape:",y.shape)
    y = MaxPooling3D((1,2, 2), strides=(1,2, 2), name='pool%d_%d'%(b,i))(y)
    return y 

def IceNet(classes=1000,input_shape =(1,30,30,3), weights_path=None, include_top=True, only_preprocess_fn_needed=False):
    if only_preprocess_fn_needed:
        return None, preprocess_input_vgg
    img_input_1 = Input(shape=input_shape,name = 'band_1')
    # img_input_2 = Input(shape=input_shape,name = 'band_2')
    # angle_input = Input(shape=[1],name='inc_angle')

    x = residual_block(img_input_1,64,(1,3,3),1,1)
    x = residual_block(x,128,(1,3,3),1,2)
    x = residual_block(x,256,(1,3,3),1,3)


    # y = residual_block(img_input_2,64,(3,3),2,1)
    # y = residual_block(y,128,(3,3),2,2)
    # y = residual_block(y,256,(3,3),2,3)


    # x = Concatenate()([x,y,angle_input])
    # x = Concatenate(axis = -1)([x,y])
    ##x = residual_block(x,512,(3,3),3,1)
    # x = residual_block(x,512,(3,3),3,2)
    ##x = residual_block(x,1024,(3,3),3,3)
    x = Conv3D(256, (1,1,1), padding='same', name='block%d_conv'%4)(x)
    #x = K.squeeze(x,1)

    x= Lambda(lambda z: K.squeeze(z,1), name='squeeze')(x)
    x = GlobalAveragePooling2D(name='avgpool3')(x)

    # x = Dense(512,name="fc1")(x)
    # x = Activation('relu', name='DenseRelu1')(x)
    # x = Dropout(0.1)(x)

    # x = Concatenate()([x,angle_input])
    # x =  L2Norm(alpha = 20, name='L2Norm_Scaled1')(x)
    # x = AlphaL2Norm(alpha = 20, name='AlphaL2Norm')(x)
    """x = Dense(256,name="fc1")(x)
    x = Activation('relu', name='DenseRelu1')(x)
    x = Dropout(0.1)(x)

    # x =  L2Norm(alpha = 20, name='L2Norm_Scaled2')(x)
    # x = L2Norm(alpha = 20, name='L2Norm_Scaled2')(x)
    # x = Dense(128,name="fc2")(x)
    # x = Activation('relu', name='DenseRelu2')(x)
    # x = Dropout(0.2)(x)

    x = Dense(classes, name='fc_pred')(x)
        # x = Activation('softmax', name='prob')(x)
    x = Activation('sigmoid', name='prob')(x)"""

    # x = Flatten()(x)
    x = Dense(16,name="fc1")(x)
    x = Activation('relu', name='DenseRelu1')(x)
    x = Dense(16,name="fc2")(x)
    x = Activation('relu', name='DenseRelu2')(x)
    x = Dense(16,name="fc3")(x)
    x = Activation('relu', name='DenseRelu3')(x)
    x = Dense(classes, activation='linear')(x)



    # model = Model([img_input_1,img_input_2,angle_input], x, name='icenet')
    model = Model(img_input_1, x, name='icenet')
    return model, preprocess_input_vgg

sessConfig = K.tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
sessConfig.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=sessConfig))

ENV_NAME = 'lineplot-v0'
gym.undo_logger_setup()

# Get the environment and extract the number of actions.
env = gym_lineplot.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print("action space:",nb_actions)
# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.

"""model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))"""

model,_ = IceNet(classes=nb_actions)

#print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_lineplot{}_weights1.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
