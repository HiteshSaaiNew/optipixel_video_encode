import cv2
import utils
import numpy as np
import torch
import time
import utils.improc
from utils.basic import print_, print_stats
from nets.pips2 import Pips
import torch.nn.functional as F
from torch import tensor, IntTensor
import imageio.v2 as imageio
import glob
import os
import saverloader
from video_to_frames import videoToFrames


class optiImage:

    def __init__(self, S, max_iters):
        self.filename = 'sample_stock_videos/room_video_8.mp4'
        self.S = S  # seqlen
        self.N = 8  # number of points per clip
        self.stride = 8  # spatial stride of the model
        self.timestride = 1  # temporal stride of the model
        self.iters = 16  # inference steps of the model
        self.image_size = (1080, 1920)  # input resolution
        self.max_iters = max_iters  # number of clips to run
        self.shuffle = False  # dataset shuffling
        self.log_freq = 1  # how often to make image summaries
        self.log_dir = './logs_demo'
        self.init_dir = './reference_model'
        self.device_ids = [0]

    def read_mp4(self, fn: str):
        # write_folder = "pips2/demo_images"o
        # if not os.path.exists(write_folder):
        #     os.makedirs(write_folder)
        vidcap = cv2.VideoCapture(fn)
        # count = 0
        frames = []
        while (vidcap.isOpened()):
            ret, frame = vidcap.read()
            if ret == False:
                break
            # cv2.imwrite(f"{write_folder}/{count}.jpg", frame)
            # count += 1
            frames.append(frame)
        vidcap.release()
        print('frames read from video: ', len(frames))
        return frames

    def read_image_data(self, filename: str):
        # read the image data from the file
        im = imageio.imread(filename)
        im = im.astype(np.uint8)
        return im

    def run_model(self, model, rgbs, S_max=128, N=64, iters=16, sw=None):
        rgbs = rgbs.cpu().float()  # B, S, C, H, W

        B, S, C, H, W = rgbs.shape
        assert (B == 1)
        print("N", N)
        # pick N points to track; we'll use a uniform grid
        N_ = np.sqrt(N).round().astype(np.int32)
        grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cpu')
        grid_y = 8 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - 16)
        grid_x = 8 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - 16)
        xy0 = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2
        _, S, C, H, W = rgbs.shape

        print("before squeeze ", xy0.shape)
        # zero-vel initops
        print("after squeeze: ", xy0.unsqueeze(1).shape)
        trajs_e = xy0.unsqueeze(1).repeat(1, S, 1, 1)
        print("trajs_e shae with unsqueeze repeat: ", trajs_e.shape)

        iter_start_time = time.time()

        preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
        trajs_e = preds[-1]
        print("Shape of trajs: ", trajs_e.shape)
        # print("trajs: ", trajs_e)

        iter_time = time.time() - iter_start_time
        print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S / iter_time))

        # rgbs_prep = utils.improc.preprocess_color(rgbs)
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
        return trajs_e

    def read_frames(self):
        # read the frames from the directory and returns the first 8 frames
        filenames = glob.glob('./demo_images_by_8/*.jpg')
        filenames = sorted(filenames)
        image_data = []
        for filename in filenames[:8]:
            im = self.read_image_data(filename)
            image_data.append(im)
        return image_data

    def transform_array(self, rgbs):
        rgbs = np.stack(rgbs, axis=0)  # S,H,W,3
        rgbs = rgbs[:, :, :, ::-1].copy()  # BGR->RGB
        rgbs = rgbs[::self.timestride]
        return rgbs

    def get_point_traject(self):

        filename = self.filename
        # rgbs = self.read_mp4(filename)
        rgbs = self.read_frames()

        rgbs = self.transform_array(rgbs)
        print(rgbs.shape)

        S_here, H, W, C = rgbs.shape

        global_step = 0

        model = Pips(stride=self.stride).cpu()
        if self.init_dir:
            _ = saverloader.load(self.init_dir, model)

        idx = list(range(0, max(S_here - self.S, 1), self.S))
        if self.max_iters:
            idx = idx[:self.max_iters]
        print('idx ', idx)

        final_traj = None
        for si in idx:
            global_step += 1

            iter_start_time = time.time()

            rgb_seq = rgbs[si:si+self.S]
            print("rgb_seq: ", rgb_seq.shape)
            rgb_seq = torch.from_numpy(rgb_seq).permute(0, 3, 1, 2).to(torch.float32)  # S,3,H,W
            rgb_seq = F.interpolate(rgb_seq, self.image_size, mode='bilinear').unsqueeze(0)  # 1,S,3,H,W

            with torch.no_grad():
                cur_traj = self.run_model(model, rgb_seq, S_max=self.S, N=self.N, iters=self.iters, sw=None)

            if final_traj is None:
                final_traj = cur_traj.clone()

            else:
                final_traj = torch.cat((final_traj, cur_traj), 1)

            break

        return final_traj

    def shift_collector(self, trajects):

        movement_tracker = []
        movement_tracker_raw = []
        for frame in range(0, len(trajects[0]) - 1):
            print(f'frame: {frame + 1} & {frame + 2}')
            point1, point2 = trajects[0][frame][4], trajects[0][frame + 1][4]
            shift = point2 - point1
            # [round(IntTensor.item(shift[1])), round(IntTensor.item(shift[0]))]
            # print(f"diff: {shift}")
            movement_tracker_raw.append(shift)
            shift_rnd = [round(IntTensor.item(shift[0])), round(IntTensor.item(shift[1]))]
            # print(f"diff rounded: ", shift_rnd)
            movement_tracker.append(shift_rnd)

        return movement_tracker, movement_tracker_raw

    def compression(self, frames_folder, movement_tracker):

        if not os.path.isdir("/Users/smanand/deeprl/optipixel_video_encode/compressed_images"):
            os.mkdir("/Users/smanand/deeprl/optipixel_video_encode/compressed_images")

        for idx in range(0, self.S - 1):
            next_image_matrix = self.read_image_data(f"{frames_folder}/frame{idx + 1}.jpg")
            shape = next_image_matrix.shape
            h, w, c = shape
            # crop_img = img[y:y+h, x:x+w]
            shift = movement_tracker[idx]
            print("shift: ", shift)
            shift[1], shift[0] = shift[0], shift[1]
            compressed_image_matrix = np.zeros(shape)
            if shift[1] >= 0 and shift[0] >= 0:
                compressed_image_matrix[:shift[0], :, :] = \
                    next_image_matrix[:shift[0], :, :]
                compressed_image_matrix[:, :shift[1], :] = \
                    next_image_matrix[:, :shift[1], :]
            elif shift[1] >= 0 and shift[0] <= 0:
                compressed_image_matrix[:shift[0], :, :] = \
                    next_image_matrix[:shift[0], :, :]
                compressed_image_matrix[:, :shift[1], :] = \
                    next_image_matrix[:, :shift[1], :]
            elif shift[1] <= 0 and shift[0] >= 0:
                compressed_image_matrix[:shift[0], :, :] = \
                    next_image_matrix[:shift[0], :, :]
                compressed_image_matrix[:, shape[1] + shift[1]:, :] = \
                    next_image_matrix[:, shape[1] + shift[1]:, :]

            elif shift[1] <= 0 and shift[0] <= 0:
                compressed_image_matrix[shape[0] + shift[0]:, :, :] = \
                    next_image_matrix[shape[0] + shift[0]:, :, :]
                compressed_image_matrix[:, shape[1] + shift[1]:, :] = \
                    next_image_matrix[:, shape[1] + shift[1]:, :]

            compressed_image_matrix = compressed_image_matrix.astype(np.uint8)
            imageio.imwrite(f'/Users/smanand/deeprl/optipixel_video_encode/compressed_images/pips2_compressed_image_{idx}.jpg', compressed_image_matrix)

    def decompress_to_frames(self, frames_folder, compress_folder, shifts):
        if not os.path.isdir("/Users/smanand/deeprl/optipixel_video_encode/decompressed_images"):
            os.mkdir("/Users/smanand/deeprl/optipixel_video_encode/decompressed_images")

        for idx in range(0, self.S - 1):

            if idx == 0:
                cur_org_frame = self.read_image_data(f"{frames_folder}/frame{idx}.jpg")
                cur_comp_frame = self.read_image_data(f"{compress_folder}/compressed_image_{idx}.jpg")

            else:
                cur_org_frame = self.read_image_data(f"/Users/smanand/deeprl/optipixel_video_encode/decompressed_images/decompressed_image_{idx}.jpg")
                cur_comp_frame = self.read_image_data(f"{compress_folder}/compressed_image_{idx}.jpg")

            shift = shifts[idx]
            h, w, _ = cur_comp_frame.shape

            if shift[0] > 0 and shift[1] < 0:
                cur_comp_frame[-shift[1]:, shift[0]:, :] = cur_org_frame[-shift[1]:, shift[0]:, :].copy()
            elif shift[0] > 0 and shift[1] > 0:
                start_p_h = h - shift[1]
                cur_comp_frame[:start_p_h, shift[0]:, :] = cur_org_frame[:start_p_h, shift[0]:, :].copy()
            elif shift[0] < 0 and shift[1] < 0:
                cur_comp_frame[-shift[1]:, :w - (-shift[0]), :] = cur_org_frame[-shift[1]:, :w - (-shift[0]), :].copy()
            elif shift[0] < 0 and shift[1] > 0:
                start_p_h = h - shift[1]
                cur_comp_frame[:start_p_h, :w - (-shift[0]), :] = cur_org_frame[:start_p_h, :w - (-shift[0]), :].copy()

            if idx == 0:
                imageio.imwrite(f'decompressed_images/pips2_decompressed_image_{idx}.jpg', cur_org_frame)

            imageio.imwrite(f'decompressed_images/pips2_decompressed_image_{idx + 1}.jpg', cur_comp_frame)

    def decompress_video(self, frames_folder, decompress_folder, shifts):
        frames = []
        for idx in range(0, self.S - 1):
            if idx == 0:
                cur_frame = self.read_image_data(f"{frames_folder}/frame{idx}.jpg")
            else:
                cur_frame = self.read_image_data(f"{decompress_folder}/decompressed_image_{idx}.jpg")

            # if idx == 0:
            #     cur_org_frame = self.read_image_data(f"{frames_folder}/frame{idx}.jpg")
            #     cur_comp_frame = self.read_image_data(f"{compress_folder}/compressed_image_{idx}.jpg")
            #     frames.append(cur_comp_frame)

            # else:
            #     cur_org_frame = self.read_image_data(f"decompressed_images/decompressed_image_{idx-1}.jpg")
            #     cur_comp_frame = self.read_image_data(f"{compress_folder}/compressed_image_{idx}.jpg")

            # shift = shifts[idx]
            # h, w , _ = cur_comp_frame.shape

            # if shift[1] > 0: 
            #     cur_comp_frame[shift[1]:, :, :] = cur_org_frame[shift[1]:, :, :]
            # if shift[1] < 0:
            #     cur_comp_frame[:h-(-shift[1]), :, :] = cur_org_frame[:h-(-shift[1]), :, :]
            # if shift[0] < 0:
            #     cur_comp_frame[ : , :w-(-shift[0]), :] = cur_org_frame[:, :w-(-shift[0]), :]
            # if shift[0] > 0:
            #     cur_comp_frame[ : , shift[0]:, :] = cur_org_frame[: , shift[0]:, :] 

            frames.append(cur_frame)

        h, w, _ = cur_frame.shape
        frames = np.stack(frames, axis=0)
        frames_rgb = frames[:, :, :, ::-1].copy()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter("decompressed_video.mp4", fourcc, 1, (w, h))

        for img in frames_rgb:
            video.write(img)

        cv2.destroyAllWindows()
        video.release()
        return

    def storage_saving_comparison(self):
        files = glob.glob("demo_images_8/*.*")
        files = sorted(files)
        total_size = 0
        for idx in range(0, 8):
            file = f"demo_images_8/frame{idx}.jpg"
            cur_img_size = os.stat(file).st_size / 1000
            total_size += cur_img_size
            if idx == 0:
                first_frame_size = cur_img_size

        print("Actual image size for 8F in KB:", total_size)
        print("Actual image size for 8F in MB:", total_size / 1000)

        files = glob.glob("compressed_images/*.*")
        total_size = 0
        for file in files:
            total_size += os.stat(file).st_size / 1000

        print("compressed image size for 8F in KB:", total_size + first_frame_size)
        print("compressed image size for 8F in MB:", (total_size + first_frame_size) / 1000)


if __name__ == '__main__':
    # videoToFrames("demo_images_2", "sample_stock_videos/room_video.mp4")
    opti_img = optiImage(8, 1)
    trajs = opti_img.get_point_traject()  ##Get trajecter for first 8 frames
    movement_tracker, movement_tracker_raw = opti_img.shift_collector(trajs)  ##Get the movement shift round/raw
    opti_img.compression("demo_images_by_30", movement_tracker)  ## Compress image based on shift of the pixel
    opti_img.decompress_to_frames("demo_images_by_30", "compressed_images", movement_tracker)
    opti_img.decompress_video("demo_images_by_30", "decompressed_images", movement_tracker)
