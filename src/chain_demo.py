import time
import numpy as np
import io
import os
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from torch import tensor, IntTensor
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from timeit import default_timer as timer

random.seed(125)
np.random.seed(125)

image_folder = "demo_images_30"


class Memory:
    def __init__(self):
        self.data = []

    def append(self, data):
        self.data.append(data)

    def get_data(self):
        return self.data


class Compreesionmatrix:
    def __init__(self):
        self.data = []

    def append(self, data):
        self.data.append(data)

    def get_data(self):
        return self.data


def run_model(model, rgbs, N):
    rgbs = rgbs.cpu().float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    print('shape of rgbs:', rgbs.shape)
    rgbs_ = rgbs.reshape(B * S, C, H, W)
    H_, W_ = 1080, 1920
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # try to pick a point on the dog, so we get an interesting trajectory
    # x = torch.randint(-10, 10, size=(1, N), device=torch.device('cuda')) + 468
    # y = torch.randint(-10, 10, size=(1, N), device=torch.device('cuda')) + 118
    x = torch.ones((1, N), device=torch.device('cpu')) * 450.0
    y = torch.ones((1, N), device=torch.device('cpu')) * 100.0
    xy0 = torch.stack([x, y], dim=-1) # B, N, 2
    _, S, C, H, W = rgbs.shape

    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cpu')
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cpu')
        traj_e[:, 0] = xy0[:, n]  # B, 1, 2  # set first position
        feat_init = None
        while not done:
            end_frame = cur_frame + 8

            rgb_seq = rgbs[:, cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:, -1].unsqueeze(1).repeat(1, 8 - S_local, 1, 1, 1)], dim=1)

            outs = model(traj_e[:, cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init,
                         return_feat=True)


            preds = outs[0]

            vis = outs[2]  # B, S, 1
            feat_init = outs[3]

            vis = torch.sigmoid(vis)  # visibility confidence
            # print('output of the model: ', preds[-1])
            xys = preds[-1].reshape(1, 8, 2)
            # print('xys from the model: ', xys)
            traj_e[:, cur_frame:end_frame] = xys[:, :S_local]

            found_skip = False
            thr = 0.9
            si_last = 8 - 1  # last frame we are willing to take
            si_earliest = 1  # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0, si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:, :, n] = traj_e
        # print('trajectory for keypoint %d: ' % n, trajs_e)

    pad = 50
    rgbs = F.pad(rgbs.reshape(B * S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H + pad * 2,
                                                                                            W + pad * 2)
    print('trajectory before pad: ', trajs_e)
    print(trajs_e[0][0][0], trajs_e[0][1][0])
    return_data = trajs_e
    trajs_e = trajs_e + pad
    # print('trajectory after pad: ', trajs_e)

    # prep_rgbs = utils.improc.preprocess_color(rgbs)
    # gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    #
    # if sw is not None and sw.save_this:
    #     linewidth = 2
    #
    #     for n in range(N):
    #         # print('visualizing kp %d' % n)
    #         kp_vis = sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n),
    #                                          trajs_e[0:1, :, n:n + 1], gray_rgbs[0:1, :S], cmap='spring',
    #                                          linewidth=linewidth)
    #
    #         # write to disk, in case that's more convenient
    #         kp_list = list(kp_vis.unbind(1))
    #         kp_list = [kp[0].permute(1, 2, 0).cpu().numpy() for kp in kp_list]
    #         kp_list = [Image.fromarray(kp) for kp in kp_list]
    #         out_fn = './chain_dog_out_%d.gif' % sw.global_step
    #         kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
    #         print('saved %s' % out_fn)
    #
    #     sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1, 0], cmap='spring')
    #     sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1),
    #                            cmap='spring')

    return tensor([[return_data[0][0][0][0], return_data[0][0][0][1]], [return_data[0][1][0][0], return_data[0][1][0][1]]])


def read_image_data(filename: str, file_index: int):
    # read the image data from the file
    im = imageio.imread(filename)
    im = im.astype(np.uint8)
    imageio.imwrite(f'./8bit_image_new/8bit_image_{file_index}.jpg', im)
    return im


def read_image_data_decompress(filename: str):
    # read the image data from the file
    im = imageio.imread(filename)
    im = im.astype(np.uint8)
    return im


def compression(filename: str, shift: tensor, file_index: int):
    print('in compression')
    print('reading file: ', filename)
    original_image_matrix = read_image_data(filename, file_index)
    print(original_image_matrix.shape)
    print('shift: ', shift)
    shape = original_image_matrix.shape
    shift = shift[1] - shift[0]
    shift = [round(IntTensor.item(shift[1])), round(IntTensor.item(shift[0]))]

    compression_start_time = timer()

    compressed_image_matrix = np.zeros(shape)
    # shift[1] -= 30
    print(shift)

    if shift[1] >= 0 and shift[0] >= 0:
        compressed_image_matrix[:shift[0], :, :] = \
            original_image_matrix[:shift[0], :, :]
        compressed_image_matrix[:, :shift[1], :] = \
            original_image_matrix[:, :shift[1], :]
    elif shift[1] >= 0 and shift[0] <= 0:
        compressed_image_matrix[shape[0] + shift[0]:, :, :] = \
            original_image_matrix[shape[0] + shift[0]:, :, :]
        compressed_image_matrix[:, :shift[1], :] = \
            original_image_matrix[:, :shift[1], :]
    elif shift[1] <= 0 and shift[0] >= 0:
        compressed_image_matrix[:shift[0], :, :] = \
            original_image_matrix[:shift[0], :, :]
        compressed_image_matrix[:, shape[1] + shift[1]:, :] = \
            original_image_matrix[:, shape[1] + shift[1]:, :]

    elif shift[1] <= 0 and shift[0] <= 0:
        compressed_image_matrix[shape[0] + shift[0]:, :, :] = \
            original_image_matrix[shape[0] + shift[0]:, :, :]
        compressed_image_matrix[:, shape[1] + shift[1]:, :] = \
            original_image_matrix[:, shape[1] + shift[1]:, :]

    compressed_image_matrix = compressed_image_matrix.astype(np.uint8)
    compression_end_time = timer()
    print('compression time: ', (compression_end_time - compression_start_time)*1000)
    # break
    imageio.imwrite(f'./compressed_image_new/compressed_image_{file_index}.jpg', compressed_image_matrix)
    return shift, compressed_image_matrix


def read_frames(number: int):
    # read the frames from the directory and returns the first 8 frames
    filenames = glob.glob(f'./{image_folder}/*.jpg')
    filenames = sorted(filenames)
    return filenames[:number]

def read_compressed_frames():
    # read the frames from the directory and returns the first 8 frames
    filenames = glob.glob('./compressed_image_new/*.jpg')
    filenames = sorted(filenames)
    return filenames[:]


def decompress(memory, compression_matrix_rgb):
    decompress_start_time = timer()
    original_frames = read_frames(1)
    print('reading frames: ', original_frames)
    memory = memory.get_data()
    original_frame = read_image_data_decompress(original_frames[0])
    for i in range(0, len(memory)):
        compressed_frame = compression_matrix_rgb[i]
        decompress_start_time = timer()
        shape = compressed_frame.shape
        print('shape: ', shape)
        decompressed_frame = compressed_frame
        shift = memory[i]
        print('shift: ', shift)
        if shift[1] >= 0 and shift[0] >= 0:
            decompressed_frame[:shape[0] - shift[0], :shape[1] - shift[1], :] = \
                original_frame[shift[0]:, shift[1]:, :]
        elif shift[1] >= 0 and shift[0] <= 0:
            decompressed_frame[-shift[0]:, :shape[1] - shift[1], :] = \
                original_frame[:shape[0] + shift[0], shift[1]:, :]
        elif shift[1] <= 0 and shift[0] >= 0:
            print('decompressing for shift[1] <= 0 and shift[0] >= 0 ')
            decompressed_frame[shift[0]:, : shape[1] + shift[1], :] = original_frame[:shape[0] - shift[0], - shift[1]:, :]
            # decompressed_frame[shift[0]:, -shift[1]: shape[1] + shift[1], :] = original_frame[shift[0]:, -shift[1]: shape[1] + shift[1], :]
        elif shift[1] <= 0 and shift[0] <= 0:
            decompressed_frame[:shape[0] + shift[0], : shape[1] + shift[1], :] = original_frame[- shift[0]:, - shift[1]:, :]
        decompressed_frame = decompressed_frame.astype(np.uint8)
        imageio.imwrite(f'./decompressed_image_new/decompressed_image_{i + 1}.jpg', decompressed_frame)
        original_frame = decompressed_frame
        decompress_end_time = timer()
        print('decompression time: ', (decompress_end_time - decompress_start_time) * 1000)
    # decompress_end_time = timer()
    # print('decompression time: ', int(decompress_end_time - decompress_start_time)*1000)


def decompress_video(frames_folder, decompress_folder):
    frames = []
    for idx in range(0, 297):
        if idx == 0:
            cur_frame = read_image_data_decompress(f"{frames_folder}/frames000001.jpg")
        else:
            cur_frame = read_image_data_decompress(f"{decompress_folder}/decompressed_image_{idx}.jpg")

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
    video = cv2.VideoWriter("decompressed_video.mp4", fourcc, 30, (w, h))

    for img in frames_rgb:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    return


def decompress_video_8bit(decompress_folder):
    frames = []
    for idx in range(0, 296):
        cur_frame = read_image_data_decompress(f"{decompress_folder}/8bit_image_{idx}.jpg")

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
    video = cv2.VideoWriter("8bit_video.mp4", fourcc, 30, (w, h))

    for img in frames_rgb:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    return


def main():
    # the idea in this file is to chain together pips from a long sequence, and return some visualizations

    exp_name = '00'  # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 2
    N = 1  # number of points to track

    filenames = glob.glob(f'./{image_folder}/*.jpg')
    filenames = sorted(filenames)
    max_iters = len(filenames) // (S // 2) - 1  # run slightly overlapping subseqs

    print('max iter ', max_iters)

    log_freq = 1  # when to produce visualizations

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    log_dir = 'logs_chain_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    model = Pips(stride=4).cpu()
    parameters = list(model.parameters())
    print(f'loading the model... at {init_dir}')
    if init_dir:
        _ = saverloader.load(init_dir, model)
    print('loaded!')
    global_step = 0
    file_index = 0
    fn = ""
    model.eval()
    memory = Memory()
    compression_matrix = []

    while global_step < max_iters:

        read_start_time = time.time()

        global_step += 1

        try:
            rgbs = []
            for s in range(S):
                fn = filenames[(global_step - 1) * S // 2 + s]
                if s == 0:
                    print('start frame', fn)
                else:
                    print('next frame', fn)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                rgbs.append(torch.from_numpy(im).permute(2, 0, 1))
            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0)  # 1, S, C, H, W

            read_time = timer() - read_start_time
            iter_start_time = timer()

            with torch.no_grad():
                model_call_time = timer()
                trajs_e = run_model(model, rgbs, N)
                model_end_time = timer()
                shift, compressed_matrix_data = compression(fn, trajs_e, file_index)
                memory.append(shift)
                compression_matrix.append(compressed_matrix_data)
                file_index += 1
                print('model time', (model_end_time-model_call_time)*1000)

            iter_time = timer() - iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)

    compression_matrix = np.stack(compression_matrix, axis=0)
    # compression_matrix_rgb = compression_matrix[:, :, :, ::-1].copy()

    decompress(memory, compression_matrix)
    decompress_video(image_folder, 'decompressed_image_new')
    decompress_video_8bit('8bit_image_new')

    writer_t.close()


if __name__ == '__main__':
    main()
    # decompress_video('demo_images_30', 'decompressed_images')
