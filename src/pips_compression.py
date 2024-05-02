import torch
import imageio.v2 as imageio
import numpy as np
import glob
from torch import tensor, IntTensor


class Memory:
    def __init__(self):
        self.data = []

    def append(self, data):
        self.data.append(data)

    def get_data(self):
        return self.data


def read_frames(number: int):
    # read the frames from the directory and returns the first 8 frames
    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    return filenames[:number]


def read_compressed_frames():
    # read the frames from the directory and returns the first 8 frames
    filenames = glob.glob('./compressed_images/*.jpg')
    filenames = sorted(filenames)
    return filenames[:]


def read_image_data(filename: str):
    # read the image data from the file
    im = imageio.imread(filename)
    im = im.astype(np.uint8)
    return im


def get_data():
    # here we will call the model on the frames and get the x and y of the trajectory
    data = tensor([[450.0000, 100.0000],
                   [330.5911,  105.5728],
                   [363.1250,  95.6724],
                   [354.9813,  90.4617],
                   [391.3014,  93.9302],
                   [382.2925,  94.1523],
                   [366.5214,  95.4221],
                   [373.6513,  94.2867]])
    return data


def compression(trajectory: list) -> int:
    return 0


def decompress(memory):
    original_frames = read_frames(1)
    compressed_frames = read_compressed_frames()
    memory = memory.get_data()
    print(memory)
    original_frame = read_image_data(original_frames[0])
    for i in range(0, len(compressed_frames)):
        compressed_frame = read_image_data(compressed_frames[i])
        shape = compressed_frame.shape
        decompressed_frame = compressed_frame
        shift = memory[i]
        print(shift)
        if shift[1] >= 0 and shift[0] >= 0:
            decompressed_frame[shift[0]:, :, :] = \
                original_frame[shift[0]:, :, :]
            decompressed_frame[:, shift[1]:, :] = \
                original_frame[:, shift[1]:, :]
        elif shift[1] >= 0 and shift[0] <= 0:
            decompressed_frame[shift[0]:, :, :] = \
                original_frame[shift[0]:, :, :]
            decompressed_frame[:, shift[1]:, :] = \
                original_frame[:, shift[1]:, :]
        elif shift[1] <= 0 and shift[0] >= 0:
            print('decompressing for shift[1] <= 0 and shift[0] >= 0 ')
            decompressed_frame[shift[0]:, : shape[1] + shift[1], :] = original_frame[:shape[0] - shift[0], - shift[1]:, :]
            # decompressed_frame[shift[0]:, -shift[1]: shape[1] + shift[1], :] = original_frame[shift[0]:, -shift[1]: shape[1] + shift[1], :]
        elif shift[1] <= 0 and shift[0] <= 0:
            decompressed_frame[:shape[0] + shift[0], : shape[1] + shift[1], :] = original_frame[- shift[0]:, - shift[1]:, :]
        decompressed_frame = decompressed_frame.astype(np.uint8)
        imageio.imwrite(f'./decompressed_images/decompressed_{i + 1}.jpg', decompressed_frame)
        break
        original_frame = decompressed_frame


def main():
    frames = read_frames(8)
    movement_data = get_data()
    memory = Memory()
    last_image_matrix = read_image_data(frames[0])
    for file_index in range(1, len(frames)):
        original_image_matrix = read_image_data(frames[file_index])
        print(original_image_matrix.shape)
        shape = original_image_matrix.shape
        shift = movement_data[file_index] - movement_data[0]
        shift = [round(IntTensor.item(shift[1])), round(IntTensor.item(shift[0]))]
        memory.append(shift)

        compressed_image_matrix = np.zeros(shape)
        # print(shift)

        if shift[1] >= 0 and shift[0] >= 0:
            compressed_image_matrix[:shift[0], :, :] = \
                original_image_matrix[:shift[0], :, :]
            compressed_image_matrix[:, :shift[1], :] = \
                original_image_matrix[:, :shift[1], :]
        elif shift[1] >= 0 and shift[0] <= 0:
            compressed_image_matrix[:shift[0], :, :] = \
                original_image_matrix[:shift[0], :, :]
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
            lost_image_part = original_image_matrix[:shape[0] + shift[0], : shape[1] + shift[1], :]

        compressed_image_matrix = compressed_image_matrix.astype(np.uint8)
        # break
        imageio.imwrite(f'./compressed_images/compressed_image_{file_index}.jpg', compressed_image_matrix)
        # imageio.imwrite(f'./compressed_images/lost_image_{file_index}.jpg', lost_image_part)
        break
    # print(memory.get_data())
    decompress(memory)
    # print(compressed_image_matrix)


# if __name__ == "__main__":
#     frames = read_frames(8)
#     image_data = read_image_data(frames[0])
#     print(image_data.shape)


def retrieve():
    frames = read_frames()
    first_image_matrix = read_image_data(frames[0])

    # retrieve the compressed images and return the compressed images

    filenames = glob.glob('./compressed_images/*.jpg')
    filenames = sorted(filenames)
    return filenames


if __name__ == "__main__":
    main()
