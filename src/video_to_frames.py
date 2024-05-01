from PIL import Image
import os
import glob
import cv2


def videoToFrames(folder_path, video_file_path):
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    vidcap = cv2.VideoCapture(video_file_path)
    success,image = vidcap.read() 
    count = 0
    while success:
        cv2.imwrite(f"{folder_path}/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1


videoToFrames("demo_images_2", "sample_stock_videos/room_video.mp4")