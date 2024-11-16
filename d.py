import cv2 as cv
import numpy as np
import argparse
import pandas as pd
#calling the links 

prototxt_path='links/colorization_deploy_v2.prototxt'
model_path='links/colorization_release_v2.caffemodel'
kernel_path='links/pts_in_hull.npy'

#inputing the image
ap=argparse.ArgumentParser() 
ap.add_argument("-i","--image",type=str,required=True,help="path to input black and white image")
args=vars(ap.parse_args())

# loading the models
print("loading model.....")
net=cv.dnn.readNetFromCaffe(prototxt_path,model_path)
points=np.load(kernel_path)


class8=net.getLayerId("class8_ab")
conv8=net.getLayerId("conv8_313_rh")
points=points.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs=[points.astype("float32")]
net.getLayer(conv8).blobs=[np.full([1,313],2.606,dtype="float32")]
  
image =cv.imread(args["image"])
scaled= image.astype("float32")/255.0
lab=cv.cvtColor(scaled,cv.COLOR_BGR2LAB)

resized=cv.resize(lab,(224,224))#same size as model is trained so taht we dont mess up size 
L=cv.split(resized)[0]
L-=50

print("colourizing image...")
net.setInput(cv.dnn.blobFromImage(L))
ab=net.forward()[0,:,:,:].transpose((1,2,0))
ab=cv.resize(ab,(image.shape[1],image.shape[0]))
L=cv.split(lab)[0]
colourised=np.concatenate((L[:,:,np.newaxis],ab),axis=2)

colourised=cv.cvtColor(colourised,cv.COLOR_LAB2BGR)
colourised=np.clip(colourised,0,1)
colourised=(255* colourised).astype("uint8")
cv.imshow("original",image)
cv.imshow("colourised",colourised)
cv.waitKey(0)
