import numpy as np
import cv2
import tensorflow as tf

def scale_to_im(x,a=0,b=255):
    """
    Normalize the image data with Min-Max scaling to a range of [a b]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    ma=(np.max(x))
    if(ma == 0):
        return x.astype(np.uint8)
    mi=(np.min(x))
    normalized_data = ((x.astype(np.float)-float(mi))/float(ma)) # normalize [0-1]
    normalized_data = (normalized_data*b + a*(1-normalized_data)) #Scale values here
    return normalized_data.astype(np.uint8)

def nothing(x):
    pass

def channels3(x):
    #Stack grayscale images together to increase the color channels to 3
    return np.dstack((x,x,x))

def sidebyside(x,y):
    #Concatenate images side by side (horizontally)
    return np.concatenate((x,y),axis=1)

def updown(x,y):
    #Concatenate images up and down (vertically)
    return np.concatenate((x,y),axis=0)

def extractLargerSegment(maskROAD):

    _, contours, hierarchy = cv2.findContours(maskROAD.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    maxA = 0
    maskTemp=np.zeros_like(maskROAD)

    if(len(contours) > 0):
        for h,cnt in enumerate(contours):
            if(cv2.contourArea(cnt) > maxA):
                cntMax=cnt
                maxA = cv2.contourArea(cnt)
        mask = np.zeros(maskROAD.shape,np.uint8)
        cv2.drawContours(maskTemp,[cntMax],0,255,-1)
        maskROAD = cv2.bitwise_and(maskROAD,maskTemp)
    return maskROAD

def post_process(img):
    kernel = np.ones((5, 5), np.uint8)
    img_out = cv2.erode(img, kernel,iterations=3)
    kernel = np.ones((20, 20), np.uint8)
    img_out = cv2.dilate(img_out, kernel,iterations=5)

    img_out = extractLargerSegment(img_out)

    return img_out


def display(img_init, img_hsv, img_out2, img_out):
    mask = scale_to_im(np.dstack((img_out, np.zeros_like(img_out), np.zeros_like(img_out))))
    cv2.imshow('Output', updown(sidebyside(cv2.addWeighted(img_init, 1, mask, 0.3, 0), img_hsv),sidebyside(channels3(img_out), channels3(img_out2))))


def detectionProcess(frame,model,winH=32,winW=32,depth=1,nb_images=2,scale=1.2,stepSize=10, thres_score = 0):
    index=0
    totalWindows = 0
    correct=0
    
    bbox_list = []
    score = []
    
    for resized in pyramid(frame, scale=scale,minSize=(winH,winW),nb_images=nb_images):
        #gray = cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY)
        # loop over the sliding window for each layer of the pyramid
        scale = frame.shape[0]/resized.shape[0]
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            
            if(depth == 1):
                window = cv2.cvtColor(window,cv2.COLOR_BGR2GRAY)
                window = np.expand_dims(window,3)
                
            window = window[None,:,:,:]
    
            totalWindows+=1
    
            class_out = model.predict((window.astype(np.float32))/255.,batch_size =1)[0]

            if(class_out < thres_score):
                bbox_list.append(((int(x*scale)),int(y*scale),int((x+winW)*scale),int((y+winH)*scale)))     
                score.append(class_out)
                correct+=1

        
        index+=1 

    
    return bbox_list,totalWindows,correct,score
    
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale=1.5, minSize=(30, 30),nb_images=3):
    # yield the original image
	yield image
	count = 0

	# keep looping over the pyramid
	while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            h = int(image.shape[0] / scale)

            image = cv2.resize(image, (w,h))
            count+=1
            scale = np.power((1/scale),count)
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0] or (count == nb_images):
                break

            # yield the next image in the pyramid
            yield image
        
def drawBoxes(frame,bbox_list):
    
    for i in range(len(bbox_list)):
        box = bbox_list[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        
    return frame
