import numpy as np
import cv2
import time
import logging
# Make sure that caffe is on the python path:
caffe_root = '../caffe-ssd/python'  # this file is expected to be in {sfd_root}/sfd_test_code/AFW

import os
import sys
sys.path.insert(0, caffe_root)
import caffe
fRegistered = 1
logging.basicConfig(filename='detectionMisses.log',level=logging.DEBUG)

if __name__ == '__main__':
        caffe.set_device(0)
        caffe.set_mode_gpu()
        model_def = 'deploy.prototxt'
        model_weights = 'SFD.caffemodel'
        net = caffe.Net(model_def, model_weights, caffe.TEST)
        
        path = '/home/xdepartment/ly/data/jdface/registered90k'
        resultPath = '/home/xdepartment/ly/data/jdface/sfd90kResult/'
        files = os.listdir(path)
        print(len(files))
        count = 0
        debug = 0
        for file in files:      
            count = count + 1
            print(count)
            print(file)
            #if file != '1958.jpg':
            #    continue
            frame = cv2.imread(path+'/'+file)
            #print(file)
            #print(frame.shape)
            #frame = cv2.resize(frame, (200,200))
            start = time.time()
            image = frame
            height = image.shape[0]
            width = image.shape[1]
            #print('width,height:'+str(width)+','+str(height))
            if max(image.shape[0], image.shape[1]) < 320 or fRegistered == 1:
                  im_shrink = 80.0 / max(image.shape[0], image.shape[1])
            else:
                  im_shrink = 320.0 / max(image.shape[0], image.shape[1])
            image = cv2.resize(image, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

            net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', np.array([104, 117, 123]))
            transformer.set_raw_scale('data', 255)
            transformer.set_channel_swap('data', (2, 1, 0))
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image

            detections = net.forward()['detection_out']
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3]
            det_ymin = detections[0, 0, :, 4]
            det_xmax = detections[0, 0, :, 5]
            det_ymax = detections[0, 0, :, 6]
            
            flag = 0
            bbox = []
            for i in range(det_conf.shape[0]):
                xmin = max(0, int(round(det_xmin[i] * width)))
                ymin = max(0, int(round(det_ymin[i] * height)))
                xmax = min(width-1, int(round(det_xmax[i] * width)))
                ymax = min(height-1, int(round(det_ymax[i] * height)))
            # simple fitting to AFW, because the gt box of training data (i.e., WIDER FACE) is longer than the gt box of AFW
            # ymin += 0.2 * (ymax - ymin + 1)   
                score = det_conf[i]
                
                if debug == 0:
                   if score <= 0.2 or xmin >= xmax or ymin >= ymax: 
                     continue
                   area = (ymax-ymin)*(xmax-xmin)
                   if area < 300:
                     continue
                else:
                   print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                   format('person', score, xmin, ymin, xmax, ymax))
                
                bbox.append([xmin, ymin, xmax, ymax, score])
                flag = 1
            end = time.time()
            print(end-start)
            
            if debug == 0:
                if flag == 0:
                    #cv2.imshow('result', frame)
                    #cv2.waitKey(0)
                    logging.debug('Missed:'+file)
                else:
                    #print(bbox)
                    tBbox = bbox[0]
                    cv2.rectangle(frame, (tBbox[0], tBbox[1]), (tBbox[2], tBbox[3]), (255,0,0), 2)
                    cv2.imwrite(resultPath+file, frame)
                    #cv2.imshow('result',frame)
                    #cv2.waitKey(0)
            else:    
                if count == 290:
                    cv2.imshow('result', frame)
                    cv2.waitKey(0)
            
            #cv2.imwrite('processed/frame{:d}.jpg'.format(count),frame)




