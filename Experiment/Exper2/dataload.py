import struct
import numpy as np
# 读取数据

def decode_idx3_ubyte(file):
    # 读取数据
    bin_data = open(file, 'rb').read()
    offset = 0  # 偏置量
    fmt_header = ">iiii"    # 表示4个整形
    # struct.unpack_from(fmt=,buffer=,offfset=) : 
    # 该函数可以将缓冲区buffer中的内容在按照指定的格式fmt='somenformat'，从偏移量为offset=numb的位置开始进行读取
    _, img_num, img_h, img_w = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据
    img_size = img_h * img_w
    offset += struct.calcsize(fmt_header)   # 计算给定的格式占用字节数，并后移偏置量
    fmt_img = ">" + str(img_size) + "B"  # 读取一张图片的字节数
    images = np.empty((img_num, 1, img_h, img_w))
    for i in range(img_num):
        images[i] = np.array(struct.unpack_from(fmt_img, bin_data, offset)).reshape((1,img_h, img_w))
        offset += struct.calcsize(fmt_header)   #每读取一张图片后，偏置量移动到下一张图片
    return images

def decode_idx1_ubyte(file):
    # 读取数据
    bin_data = open(file, 'rb').read()
    offset = 0  # 偏置量
    fmt_header = ">ii"    # 表示2个整形
    _, lab_num = struct.unpack_from(fmt_header, bin_data, offset)
    
    # 解析数据
    offset += struct.calcsize(fmt_header)
    fmt_lab = ">B"
    labels = np.empty((lab_num))
    for i in range(lab_num):
        labels[i] = np.array(struct.unpack_from(fmt_lab, bin_data, offset))
        offset += struct.calcsize(fmt_lab)
    return labels

# decode_idx3_ubyte("./data/train-images.idx3-ubyte")
# decode_idx1_ubyte("./data/train-labels.idx1-ubyte")