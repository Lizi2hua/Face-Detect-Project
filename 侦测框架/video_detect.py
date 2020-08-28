from 侦测框架.core import detector
import os
from PIL import  Image
from 侦测框架.NNTools.NMS import NMS

import cv2
import time
if __name__ == '__main__':
    video_path=r"C:\Users\liewei\Desktop\tmr.mp4"
    detector = detector.Detector()
    video = cv2.VideoCapture(video_path)
    frame_count=0
    start_time = time.time()
    while True:
        ret,frame=video.read()
        frame_count+=1
        if ret:
            b, g, r = cv2.split(frame)
            img1 = cv2.merge([r, g, b])#opencv用的图片格式是BGR，框架用的是RGB，注意数据分开
            img=Image.fromarray(img1)#w,h=img.size这儿会报错，故加这一行
            boxes=detector.detect(img)
            # boxes=NMS(boxes)
            # boxes=opt_detector.detect(im)
            if boxes.shape[0] !=0:
                for box in boxes:
                    x1=int(box[1])
                    y1=int(box[2])
                    w=int(box[3]-x1)
                    h=int(box[4]-y1)
                    squre=min(w,h)
                    # 参数根据分辨率调整
                    cv2.rectangle(frame,(x1+10,y1+10),(x1+squre,int(y1+squre)),(0,255,0),thickness=1)
                    img_crop=frame[y1+11:y1+squre,x1+11:x1+squre]
                    # cv2.imshow("face_window", img_crop)
                    # print(img_crop.shape)
                    w,h,c=img_crop.shape
                    if w<80 :
                        pass
                    else:
                        file_name=r'C:\Users\liewei\Desktop\face_data\3\{}.jpg'.format(frame_count)
                        cv2.imwrite(file_name,img_crop)
            else:
                pass
            # if boxes
            # cv2.imshow("detect_face",frame)
            cv2.imshow("detected_face", frame)

            cv2.waitKey(21)

        else:
            print("videos ended or failed to read!")
            break
    end_time=time.time()
    cost=end_time-start_time
    cv2.destroyAllWindows()
    print('==================================')
    print("avg_FPS:",frame_count/cost)

