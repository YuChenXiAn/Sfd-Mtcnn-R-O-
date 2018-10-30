import numpy as np
import cv2
import time

def sfd_detection(net, frame):

        #start = time.time()
        image = frame
        heigh = image.shape[0]
        width = image.shape[1]
        
        if max(image.shape[0], image.shape[1]) < 320:
              im_shrink = 160.0 / max(image.shape[0], image.shape[1])
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
        bbox = []
        for i in range(det_conf.shape[0]):
            xmin = int(round(det_xmin[i] * width))
            ymin = int(round(det_ymin[i] * heigh))
            xmax = int(round(det_xmax[i] * width))
            ymax = int(round(det_ymax[i] * heigh))  
            score = det_conf[i]
            if score < 0.5 or xmin < 0 or ymin < 0 or xmax >= width or ymax >= heigh:
              continue
            #print('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
            # format('person', score, xmin, ymin, xmax, ymax))
            bbox.append([xmin,ymin,xmax,ymax])
            #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        #end = time.time()
        #print(end-start)
        #cv2.imshow('result', frame)
        #cv2.waitKey(0)
        return bbox
        

