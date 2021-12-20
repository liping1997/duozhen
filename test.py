"""General-purpose train script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to train the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python train.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model train --no_dropout

    The option '--model train' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more train options.
See training and train tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from crop import cutimg
from merge import ddd,fff,ggg,eee,imgFusion
import cv2
import numpy as np


if __name__ == '__main__':
    # """
    # 下面这段代码主要用于裁剪图片，生成用于测试所需要的图片,保存在datasets/aaaaa/test文件夹下      hstack水平拼接 vstack垂直拼接
    # """
    # dir_num=len(os.listdir('./test'))
    # for k in range(dir_num):
    #     img = cv2.imread('./test/{}.jpg'.format(k))
    #     a, b, c, d = cutimg(img,49)
    #     for i in range(7):
    #         for j in range(7):
    #             img1 = np.hstack((a[i][j], b[i][j], c[i][j], d[i][j]))
    #             cv2.imwrite('./datasets/aaaaa/test/{}_{}_{}.jpg'.format(k,i,j), img1,[int(cv2.IMWRITE_JPEG_QUALITY), 100])


    opt = TestOptions().parse()  # get train options
    # hard-code some parameters for train
    opt.num_threads = 0   # train code only supports num_threads = 1
    opt.batch_size = 1    # train code only supports batch_size = 1
    opt.serial_batches = False  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the train code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print(1)
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # train with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

    # """
    # 下面这段代码用于将测试生成的图片拼接成一张完整的图片,保存在save文件夹中
    # """
    # dat_list1 = []
    # dat_list2 = []
    # dat_list3 = []
    # for k in range(dir_num):
    #     for i in range(7):
    #         for j in range(7):
    #             img1 = cv2.imread('./results/FA_sequence/test_latest/images/{}_{}_{}_{}.png'.format(k,i, j, 'fake_B0'))
    #             img2 = cv2.imread('./results/FA_sequence/test_latest/images/{}_{}_{}_{}.png'.format(k,i, j, 'fake_B1'))
    #             img3 = cv2.imread('./results/FA_sequence/test_latest/images/{}_{}_{}_{}.png'.format(k,i, j, 'fake_B2'))
    #             dat_list1.append(img1)
    #             dat_list2.append(img2)
    #             dat_list3.append(img3)
    # for i in range(dir_num):
    #     h1 = imgFusion(dat_list1[0+49*i:7+49*i])
    #     h2 = imgFusion(dat_list1[7+49*i:14+49*i])
    #     h3 = imgFusion(dat_list1[14+49*i:21+49*i])
    #     h4 = imgFusion(dat_list1[21+49*i:28+49*i])
    #     h5 = imgFusion(dat_list1[28+49*i:35+49*i])
    #     h6 = imgFusion(dat_list1[35+49*i:42+49*i])
    #     h7 = imgFusion(dat_list1[42+49*i:49+49*i])
    #     hh1=[h1,h2,h3,h4,h5,h6,h7]
    #     dat1=imgFusion(hh1,64,False)
    #     h8 = imgFusion(dat_list2[0+49*i:7+49*i])
    #     h9 = imgFusion(dat_list2[7+49*i:14+49*i])
    #     h10 = imgFusion(dat_list2[14+49*i:21+49*i])
    #     h11 = imgFusion(dat_list2[21+49*i:28+49*i])
    #     h12 = imgFusion(dat_list2[28+49*i:35+49*i])
    #     h13 = imgFusion(dat_list2[35+49*i:42+49*i])
    #     h14 = imgFusion(dat_list2[42+49*i:49+49*i])
    #     hh2 = [h8,h9,h10,h11,h12,h13,h14]
    #     dat2 = imgFusion(hh2, 64, False)
    #     h15 = imgFusion(dat_list3[0+49*i:7+49*i])
    #     h16 = imgFusion(dat_list3[7+49*i:14+49*i])
    #     h17 = imgFusion(dat_list3[14+49*i:21+49*i])
    #     h18 = imgFusion(dat_list3[21+49*i:28+49*i])
    #     h19 = imgFusion(dat_list3[28+49*i:35+49*i])
    #     h20 = imgFusion(dat_list3[35+49*i:42+49*i])
    #     h21 = imgFusion(dat_list3[42+49*i:49+49*i])
    #     hh3 = [h15,h16,h17,h18,h19,h20,h21]
    #     dat3 = imgFusion(hh3, 64, False)
    #
    #     dat4=np.zeros((512,512,3))
    #     dat4=np.uint8(dat4)
    #     dat=np.hstack((dat4,dat1,dat2,dat3))
    #     img4=cv2.imread('./test/{}.jpg'.format(i))
    #     print(dat.shape,img4.shape)
    #     dat=np.vstack((img4,dat))
    #
    #
    #     cv2.imwrite('save/{}.jpg'.format(i), dat,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #     print("第{}张图片已经保存".format(i))

