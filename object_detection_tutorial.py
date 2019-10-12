
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import serial


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util

from utils import visualization_utils as vis_util

# In[ ]
# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
###def goToArduino():
 #   ser=serial.Serial('COM3',9600)
  #  while 1:
   #     val=lane1
    #    ser.write(val.encode())
# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')


# ## Load a (frozen) Tensorflow model into memory.
    
# In[ ]:
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code
# In[ ]:
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 21) ]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
# In[ ]:
# Running the tensorflow session
count=0
l1, l2, l3, l4, l5, l6, l7, l8, l9 = 0, 0, 0, 0, 0, 0, 0, 0, 0
l10, l11, l12, l13, l14, l15, l16, l17, l18 = 0, 0, 0, 0, 0, 0, 0, 0, 0
count1=0
ser=serial.Serial('COM3',9600)
khedunumx=[]

def funp(p,q,r,s):
    if (p>q+r+s) or (p>2*q) and (p >2*r) and (p>2*s):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif p in range(q-20,q+20) and range(r-20,r+20) and range(s-20,s+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif p in range(r-20,r+20) and p+r > q+s :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3")    
def funq(p,q,r,s):
    if q>p+r+s or (q>2*p) and (q >2*r) and (q>2*s):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif q in range(p-20,p+20) and range(r-20,r+20) and range(s-20,s+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif q in range(s-20,s+20) and q+s > p+r :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3")    
def funr(p,q,r,s):
    if r>q+p+s or (r>2*q) and (r >2*p) and (p>2*s):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif r in range(q-20,q+20) and range(p-20,p+20) and range(s-20,s+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif r in range(p-20,p+20) and p+r > q+s :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3") 
def funs(p,q,r,s):
    if s>q+r+p or (s>2*q) and (s >2*r) and (s>2*p):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif s in range(q-20,q+20) and range(r-20,r+20) and range(p-20,p+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif s in range(q-20,q+20) and s+q > p+r :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3")
'''nhedunumy1=list(map(int,khedunumx [4:8]))
p1,q1,r1,s1=nhedunumy1

def funp1(p1,q1,r1,s1):
    if p1>q1+r1+s1 or (p1>2*q1) and (p1 >2*r1) and (p1>2*s1):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif p1 in range(q1-20,q1+20) and range(r1-20,r1+20) and range(s1-20,s1+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif p1 in range(r1-20,r1+20) and p1+r1 > q1+s1 :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3")    
def funq1(p1,q1,r1,s1):
    if q1>p1+r1+s1 or (q1>2*p1) and (q1 >2*r1) and (q1>2*s1):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif q1 in range(p1-20,p1+20) and range(r1-20,r1+20) and range(s1-20,s1+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif q1 in range(s1-20,s1+20) and q1+s1 > p1+r1 :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3")    
def funr1(p1,q1,r1,s1):
    if r1>q1+p1+s1 or (r1>2*q1) and (r1 >2*p1) and (p1>2*s1):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif r1 in range(q1-20,q1+20) and range(p1-20,p1+20) and range(s1-20,s1+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif r1 in range(p1-20,p1+20) and p1+r1 > q1+s1:
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3") 
def funs1(p1,q1,r1,s1):
    if s1>q1+r1+p1 or (s1>2*q1) and (s1 >2*r1) and (s1>2*p1):
        model="SLR"
        chal2= "K"
        ser.write(chal2.encode())
        
        print("huahua0")
    elif s1 in range(q1-20,q1+20) and range(r1-20,r1+20) and range(p1-20,p1+20) :
        model="SLRL"
        chal2= "L"
        ser.write(chal2.encode())
        print("huahua1")
    elif s1 in range(q1-20,q1+20) and s1+q1 > p1+r1 :
        model="SL"
        chal2= "M"
        ser.write(chal2.encode())
        print("huahua2")
    else:
        model="SLRL"
        chal2= "N"
        ser.write(chal2.encode())
        print("huahua3")'''
        
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      print ("boxes: ", (num_detections - 1)) 
      khedunum = num_detections -1
      khedunumx.append((str(khedunum)[1:-2]))
      
      
      print(str(khedunum)[1:-2])
      count+=1

      if(count == 1):
          lane1 = khedunum
          print(str(lane1)[1:-2])
          val=str(lane1)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
              
          chal = "A" 
          l1 = chal
          l2 = chal1
          l7 = val1
          #ser.write(chal.encode())
         # ser.write(chal1.encode())
          print(chal1.encode())
          
      elif(count == 2):
          lane2 = khedunum
          print(str(lane2)[1:-2])
          val=str(lane2)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
              
          chal = "B" 
          l3 = chal
          l4 = chal1
          l8 = val1
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())
          
      elif(count == 3):
          lane3 = khedunum
          print(str(lane3)[1:-2])
          val=str(lane3)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
          
          chal = "C" 
          l5 = chal
          l6 = chal1
          l9 = val1
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())
          
          
      elif(count == 4):
          lane4 = khedunum
          print(str(lane4)[1:-2])
          val=str(lane4)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
          
          chal = "D" 
          ser.write(l1.encode())
          ser.write(l2.encode())
          funp(l7,l8,l9,val1)
          ser.write(l3.encode())
          ser.write(l4.encode())
          funq(l7,l8,l9,val1)
          ser.write(l5.encode())
          ser.write(l6.encode())
          funr(l7,l8,l9,val1)
          ser.write(chal.encode())
          ser.write(chal1.encode())
          funs(l7,l8,l9,val1)
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())
          count = 0
          
          
      '''elif(count == 5):
          lane1 = khedunum
          print(str(lane1)[1:-2])
          val=str(lane1)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
          
          chal = "A"
          l10 = chal
          l11 = chal1
          l12= val1
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())
          
          
      elif(count == 6):
          lane2 = khedunum
          print(str(lane2)[1:-2])
          val=str(lane2)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
          
          chal = "B"
          l13 = chal
          l14 = chal1
          l15= val1
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())
          
          
      elif(count == 7):
          lane3 = khedunum
          print(str(lane3)[1:-2])
          val=str(lane3)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
          
          chal = "C"
          l16 = chal
          l17 = chal1
          l18= val1
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())
          
          
      elif(count == 8):
          lane4 = khedunum
          print(str(lane4)[1:-2])
          val=str(lane4)[1:-2]
          
          val1=int(val)
          print(val1 + 10)
          chal1 = ""
          if val1 in range (0, 11):
              chal1 = "E"
          elif val1 in range (11, 21):
              chal1 = "F"
          elif val1 in range (21, 31):
              chal1 = "G"
          elif val1 in range (31, 41):
              chal1 = "H"
          elif val1 in range (41, 51):
              chal1 = "I"    
          elif val1 in range (51, 61):
              chal1 = "J"
          
          chal = "D" 
          ser.write(l10.encode())
          ser.write(l11.encode())
          funp1(l12,l15,l18,val1)
          ser.write(l13.encode())
          ser.write(l14.encode())
          funq1(l12,l15,l18,val1)
          ser.write(l16.encode())
          ser.write(l17.encode())
          funr1(l12,l15,l18,val1)
          ser.write(chal.encode())
          ser.write(chal1.encode())
          funs1(l12,l15,l18,val1)
          #ser.write(chal.encode())
          #ser.write(chal1.encode())
          print(chal1.encode())'''
      
      
#print('khedunumx',khedunumx [0:4])     #nhedunumy=list(map(int,khedunumx [0:4]))
#p,q,r,s=nhedunumy

        
'''for i in range(0,8):
    count1+=1        
    if count1 == 1:
        funp()  
    elif count1 == 2:
        funq()
    elif count1 == 3:
        funr()
    elif count1 == 4:
        funs()
    elif count1 == 5:
        funp1()
    elif count1 == 6:
        funq1()
    elif count1 == 7:
        funr1()
    elif count1 == 8:
        funs1()    '''
    
     
      
        
    
    
    
    
#goToArduino()