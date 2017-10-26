import tensorflow as tf
import numpy as np
import PIL.Image
from io import BytesIO
from pylab import *
import matplotlib.animation as animation
import cv2



N = 500#the width and length of the pic
frames = 1000#total_frames
fps= 120
filename = 'demoVideo.avi'

sess = tf.InteractiveSession()


def make_kernel(a):

  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)
  #首先将矩阵转换为卷积核，其shape的前两个为卷积核的长度和宽度，后两个分别是图片输入的通道数以及处理之后输出的通道数（也是卷积核的数目）
  #卷积核的数目为1，表明直接卷积，直接返回卷积后的图像（如果为n，就相当于返回n张图像，因为这里是定值，所以返回的图像都一样，只需要一个卷积核就够了）
  #之后reshape一下，返回这个卷积核（在图像处理中，卷积核是作为变量使用，但这里是constant）



def simple_conv(x, k):#进行卷积操作
  """A simplified 2D convolution operation"""
  print(x)
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)#将shape为(x)的矩阵转换为(1,x,1)，此处相当于为(1,100,100,1)
  #第一个1是图片数目，第四个1是图片的通道数目
  print("afterchange:",x)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')

  #对图像进行卷积，实际上是过滤，每个卷积核与图像中同样大小的矩阵相乘，求得数量积，然后卷积核右移1个单位......
  #直到全部完成，将返回与原图同样大小的图片



  return y[0, :, :, 0]



def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)
#对图像x进行卷积


# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([N, N], dtype="float32")
ut_init = np.zeros([N, N], dtype="float32")

# Some rain drops hit a pond at random points

for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()
  #40个随机点




# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
#eps是时间间隔，damping是波纹强度
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))
#更新U和Ut的值

# Initialize state to initial conditions
tf.initialize_all_variables().run()

# Run 1000 steps of PDE

video_data=[]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(filename=filename,fourcc = fourcc,fps = fps,frameSize = (N,N),isColor = False)#注意如果iscolor不是False，就会要求3通道


temp = frames/100
def update():
  for i in range(frames):
    # Step simulation
    step.run({eps: 0.03, damping: 0.04})

    if i % temp ==0:
      print("{}%".format(i/temp))
    # Visualize every 50 steps


      #clear_output()
    a=U.eval()
    rng=[-0.1, 0.1]
    a = (a - rng[0]) / float(rng[1] - rng[0]) * 255
    a = np.uint8(np.clip(a, 0, 255))



    videoWriter.write(a)
  videoWriter.release()


update()









