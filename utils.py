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
    #maskROAD = cv2.dilate(maskROAD, np.ones((7, 7)))
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
    #cv2.namedWindow('Detections',0)
    
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
            
            # since we do not have a classifier, we'll just draw the window
            if(depth == 1):
                window = cv2.cvtColor(window,cv2.COLOR_BGR2GRAY)
                window = np.expand_dims(window,3)
                
            window = window[None,:,:,:]
    
            totalWindows+=1
    
            class_out = model.predict((window.astype(np.float32))/255.,batch_size =1)[0]
            #print(class_out)
            #ind = np.argmax(class_out)
    
            #if(ind == 0 and class_out[0][ind] > thres_score):
            if(class_out < thres_score):
                #cv2.rectangle(resized, (x, y), (x + winW, y + winH), (255, 255, 0), 2)
                bbox_list.append(((int(x*scale)),int(y*scale),int((x+winW)*scale),int((y+winH)*scale)))     
                score.append(class_out)
                correct+=1
            #else:
            #    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 1)
            #plt.imshow(resized)
        
        index+=1 
    #plt.show()
    
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

class yolo_params():

    def __init__(self,LABELS,COLORS,NORM_H,NORM_W,GRID_H,GRID_W,BOX,CLASS,CLASS_WEIGHTS,THRESHOLD,ANCHORS,SCALE_NOOB,SCALE_CONF,SCALE_COOR,SCALE_PROB):
        self.LABELS = LABELS
        self.COLORS = COLORS
        self.NORM_H = NORM_H
        self.NORM_W = NORM_W
        self.GRID_H = GRID_H
        self.GRID_W = GRID_W
        self.N_GRID_H, self.N_GRID_W = int(self.NORM_H / self.GRID_H), int(self.NORM_W / self.GRID_W)
        self.BOX = BOX
        self.CLASS = CLASS
        self.CLASS_WEIGHTS = CLASS_WEIGHTS
        self.FILTER = (self.CLASS + 5) * self.BOX
        self.THRESHOLD = THRESHOLD
        self.ANCHORS = ANCHORS
        self.SCALE_NOOB = SCALE_NOOB
        self.SCALE_CONF = SCALE_CONF
        self.SCALE_COOR = SCALE_COOR
        self.SCALE_PROB = SCALE_PROB

        if(len(ANCHORS)/2 != BOX):
            error('Anchor boxes length <ANCHOR> is not equal to number of boxes defined in <BOX>')

    def print_params(self):

        print('LABELS',self.LABELS)
        print('COLORS',self.COLORS)
        print('NORM_H',self.NORM_H)
        print('NORM_W',self.NORM_W)
        print('GRID_H',self.GRID_H)
        print('GRID_W',self.GRID_W)
        print('N_GRID_H',self.N_GRID_H)
        print('N_GRID_W',self.N_GRID_W)
        print('BOX',self.BOX)
        print('CLASS',self.CLASS)
        print('CLASS_WEIGHTS',self.CLASS_WEIGHTS)
        print('FILTER',self.FILTER)
        print('THRESHOLD',self.THRESHOLD)
        print('ANCHORS',self.ANCHORS)
        print('SCALE_NOOB',self.SCALE_NOOB)
        print('SCALE_CONF',self.SCALE_CONF)
        print('SCALE_COOR',self.SCALE_COOR)
        print('SCALE_PROB',self.SCALE_PROB)

    def save_params(self,name):
        m = {}
        m["yolo_params"] = self.__dict__

        file = open("./Models/" + name + "_yolo_params.pkl", "wb")
        pickle.dump(m, file)
        file.close()

    def load_params(self, name):
        return pickle.load( open( "./Models/" + name + "_yolo_params.pkl", "rb" ) )


class yLoss:

    def __init__(self,params):
        self.BOX=params.BOX
        self.ANCHORS=params.ANCHORS
        self.GRID_H=params.GRID_H
        self.GRID_W=params.GRID_W
        self.NORM_W=params.NORM_W
        self.NORM_H=params.NORM_H
        self.SCALE_CONF=params.SCALE_CONF
        self.SCALE_NOOB=params.SCALE_NOOB
        self.SCALE_COOR=params.SCALE_COOR
        self.SCALE_PROB=params.SCALE_PROB
        self.CLASS=params.CLASS
        self.CLASS_WEIGHTS=params.CLASS_WEIGHTS


    def yolo_loss(self,y_true, y_pred):
        ### Adjust prediction
        # adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[:, :, :, :, :2])

        # adjust w and h

        pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(self.ANCHORS, [1, 1, 1, self.BOX, 2])
        pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(self.GRID_W), float(self.GRID_H)], [1, 1, 1, 1, 2]))

        # adjust confidence
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)

        # adjust probability
        pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

        y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)

        ### Adjust ground truth
        # adjust x and y
        center_xy = .5 * (y_true[:, :, :, :, 0:2] + y_true[:, :, :, :, 2:4])
        center_xy = center_xy / np.reshape([(float(self.NORM_W) / self.GRID_W), (float(self.NORM_H) / self.GRID_H)], [1, 1, 1, 1, 2])
        true_box_xy = center_xy - tf.floor(center_xy)

        # adjust w and h
        true_box_wh = (y_true[:, :, :, :, 2:4] - y_true[:, :, :, :, 0:2])
        true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(self.NORM_W), float(self.NORM_H)], [1, 1, 1, 1, 2]))

        # adjust confidence
        pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([self.GRID_W, self.GRID_H], [1, 1, 1, 1, 2])
        pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
        pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
        pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

        true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([self.GRID_W, self.GRID_H], [1, 1, 1, 1, 2])
        true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
        true_box_ul = true_box_xy - 0.5 * true_tem_wh
        true_box_bd = true_box_xy + 0.5 * true_tem_wh

        intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
        intersect_br = tf.minimum(pred_box_bd, true_box_bd)
        intersect_wh = intersect_br - intersect_ul
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

        iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
        best_box = tf.to_float(best_box)
        true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

        # adjust confidence
        true_box_prob = y_true[:, :, :, :, 5:]

        y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
        # y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)

        ### Compute the weights
        weight_coor = tf.concat(4 * [true_box_conf], 4)
        weight_coor = self.SCALE_COOR * weight_coor

        weight_conf = self.SCALE_NOOB * (1. - true_box_conf) + self.SCALE_CONF * true_box_conf

        weight_prob = tf.concat(self.CLASS * [true_box_conf], 4)
        weight_prob = self.SCALE_PROB * weight_prob

        weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)

        ### Finalize the loss
        loss = tf.pow(y_pred - y_true, 2)
        loss = loss * weight
        loss = tf.reshape(loss, [-1, self.GRID_W * self.GRID_H * self.BOX * (4 + 1 + self.CLASS)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)

        return loss
