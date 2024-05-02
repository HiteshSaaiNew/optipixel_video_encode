import os
import glob

fps = 8 
clip = "stock_videos/room_video.mp4"
clip_name = "stock_videos/room_video_{8}.mp4"

command = "ffmpeg -i {0} -filter:v fps={2} {1}".format(clip, clip_name, fps)
