import numpy as np
import caffe
import cv2
from data import colorize_image as CI
import glob
import matlab.engine
import os
from skimage import color, io
import matplotlib.pyplot as plt
import pymeanshift as pms
from skimage.segmentation import slic, mark_boundaries
import time

def put_point(input_ab, mask, loc, p, val):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       half-patch size
    # val         2 tuple      (a,b) value of user input
    input_ab[:, loc[0] - p:loc[0] + p + 1, loc[1] - p:loc[1] + p + 1] = np.array(val)[:, np.newaxis, np.newaxis]
    mask[:, loc[0] - p:loc[0] + p + 1, loc[1] - p:loc[1] + p + 1] = 1
    return (input_ab, mask)

def get_global_histogram(ref_path):
    ref_img_fullres = caffe.io.load_image(ref_path)
    img_glob_dist = (255*caffe.io.resize_image(ref_img_fullres,(Xd,Xd))).astype('uint8') # load image
    gt_glob_net.blobs['img_bgr'].data[...] = img_glob_dist[:,:,::-1].transpose((2,0,1)) # put into
    gt_glob_net.forward();
    glob_dist_in = gt_glob_net.blobs['gt_glob_ab_313_drop'].data[0,:-1,0,0].copy()
    return (glob_dist_in,ref_img_fullres)


def video_to_frames(video_path, color_frames_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imshow('window-name', frame)
        cv2.imwrite(color_frames_path + "frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return fps


def make_bw_frames(color_frames_path, bw_frames_path):
    for frame_name in glob.glob(color_frames_path + "*.jpg"):
        frame = cv2.imread(frame_name, 0)
        start = frame_name.find(color_frames_path)
        cv2.imwrite(bw_frames_path + frame_name[start + len(color_frames_path):], frame)


def vanilla_network_colorize(prototxt_path, caffemodel_path, bw_frames_path, out_frames_path):
    img = cv2.imread(bw_frames_path + "frame0.jpg")
    height, width, _ = img.shape

    gpu_id = -1

    colorModel = CI.ColorizeImageCaffe(Xd)

    colorModel.prep_net(gpu_id, prototxt_path, caffemodel_path)

    mask = np.zeros((1, Xd, Xd))
    input_ab = np.zeros((2, Xd, Xd))

    for i in range(150):
        print(i)
        colorModel.load_image(bw_frames_path + "frame" + str(i) + ".jpg")

        img_out = colorModel.net_forward(input_ab, mask)
        img_out_fullres = colorModel.get_img_fullres()

        cv2.imwrite(out_frames_path + 'frame' + str(i) + '.jpg', img_out_fullres)


def feature_map_network_colorize(prototxt_path, caffemodel_path, bw_frames_path, out_frames_path):
    cid = CI.ColorizeImageCaffe(Xd=256)
    cid.prep_net(-1, prototxt_path, caffemodel_path)

    ref = cv2.imread("color_frames/frame0.jpg")
    cv2.imwrite("../colorization/Code/Input/ref.jpg", ref)

    engine = matlab.engine.start_matlab()
    engine.addpath("~/Documents/DL/video-colorization/project_code/colorization/Code", nargout=0)
    engine.addpath("~/Documents/DL/video-colorization/project_code/colorization/knn", nargout=0)
    engine.addpath("~/Documents/DL/video-colorization/project_code/colorization/edison_matlab_interface", nargout=0)
    engine.addpath("~/Documents/DL/video-colorization/project_code/colorization/TurboPixels", nargout=0)
    engine.addpath("~/Documents/DL/video-colorization/project_code/colorization/TurboPixels/lsmlib", nargout=0)
    engine.addpath("~/Documents/DL/video-colorization/project_code/colorization/OpenSURF/SubFunctions", nargout=0)

    for i in range(150):
        img_path = bw_frames_path + 'frame' + str(i) + '.jpg'
        cid.load_image(img_path)

        print("frame" + str(i))

        # if i > 0:
        #     ref_path = bw_frames_path + 'frame' + str(i-1) + '.jpg'
        # else:
        #     ref_path = bw_frames_path + 'frame0.jpg'
        #
        # img1 = cv2.imread(ref_path)
        # img2 = cv2.imread(img_path)
        #
        # orb = cv2.ORB_create()
        # kp1, des1 = orb.detectAndCompute(img1, None)
        # kp2, des2 = orb.detectAndCompute(img2, None)
        #
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        #
        # if i > 0:
        #     ref_path = out_frames_path + 'frame' + str(i-1) + '.jpg'
        # else:
        #     ref_path = 'color_frames/frame0.jpg'
        # temp = io.imread(ref_path)
        # temp = color.rgb2lab(temp)
        #
        # pix_idxs = []
        # pix_ab_vals = []
        #
        # for mat in matches[:10]:
        #     col, row = kp1[mat.queryIdx].pt
        #     row = int(row)
        #     col = int(col)
        #     pix_idxs.append((row, col))
        #     pix_ab_vals.append(temp[row][col])

        engine.Main_Colorization('frame' + str(i), 'ref', nargout=0)

        if i > 0:
            temp = io.imread('../colorization/Code/Results/frame' + str(i-1) + '_Result_sat.png')
        else:
            temp = io.imread("../colorization/Code/Input/ref.jpg")
        # if len(temp[0][0]) != 3:
        #     temp = temp[:, :, :3]
        temp = color.rgb2lab(temp)

        rgb = io.imread('../colorization/Code/Results/frame' + str(i) + '_pixmap_sat.png')
        lab = color.rgb2lab(rgb)

        pix_idxs = []
        pix_ab_vals = []

        for row in range(len(lab)):
            for col in range(len(lab[0])):
                if list(lab[row][col]) != [0, 0, 0]:
                    pix_idxs.append((row, col))
                    pix_ab_vals.append(temp[row][col])

        input_ab = np.zeros((2, 256, 256))
        mask = np.zeros((1, 256, 256))

        print(len(pix_idxs))
        for j in range(len(pix_idxs)):
            (input_ab, mask) = put_point(input_ab, mask, [pix_idxs[j][0], pix_idxs[j][1]], 2,
                                         [pix_ab_vals[j][1], pix_ab_vals[j][2]])

        img_out = cid.net_forward(input_ab, mask)
        img_out_fullres = cid.get_img_fullres()
        img_out = cv2.cvtColor(img_out_fullres, cv2.COLOR_RGB2BGR)

        # temp = cv2.imread('../colorization/Code/Results/frame' + str(i) + '_Result_sat.png')
        # cv2.imwrite("../colorization/Code/Input/ref.jpg", temp)
        cv2.imwrite(out_frames_path + 'frame' + str(i) + '.jpg', img_out)



if __name__ == '__main__':
    color_frames_path = "color_frames/"
    bw_frames_path = "bw_frames/"
    out_frames_path = "out_frames_features/"
    out_frames_vanilla_path = "out_frames_vanilla/"
    video_path = "YUP++/camera_moving/Ocean/Ocean_moving_cam_26.mp4"

    # vanilla mode
    prototxt_path = 'models/reference_model/deploy_nodist.prototxt'
    caffemodel_path = 'models/reference_model/model.caffemodel'


    Xd = 256

    # fps = video_to_frames(video_path, color_frames_path)
    # make_bw_frames(color_frames_path, bw_frames_path)
    fps = 30

    # vanilla_network_colorize(prototxt_path, caffemodel_path, bw_frames_path, out_frames_vanilla_path)
    # feature_map_network_colorize(prototxt_path, caffemodel_path, bw_frames_path, out_frames_path)

    # # img = cv2.imread(out_frames_path + "frame0.jpg")
    # img = cv2.imread('../colorization/Code/Results/frame0_Result_sat.png')
    # height, width, _ = img.shape
    # video = cv2.VideoWriter('gupta_6.mp4', -1, 30, (width, height))
    #
    # for i in range(150):
    #     # img = cv2.imread(out_frames_path + "frame" + str(i) + ".jpg")
    #     img = cv2.imread('../colorization/Code/Results/frame' + str(i) + '_Result_sat.png')
    #     video.write(img)
    #
    # cv2.destroyAllWindows()
    # video.release()


    # cid = CI.ColorizeImageCaffeGlobDist(Xd)
    # cid.prep_net(-1, prototxt_path='./models/global_model/deploy_nodist.prototxt',
    #              caffemodel_path='./models/global_model/global_model.caffemodel')
    #
    # # Global distribution network - extracts global color statistics from an image
    # gt_glob_net = caffe.Net('./models/global_model/global_stats.prototxt',
    #                         './models/global_model/dummy.caffemodel', caffe.TEST)
    #
    # ref = cv2.imread("color_frames/frame0.jpg")
    # cv2.imwrite("../colorization/Code/Input/ref.jpg", ref)
    #
    # for i in range(125):
    #     img_path = bw_frames_path + 'frame' + str(i) + '.jpg'
    #     cid.load_image(img_path)
    #
    #     print("frame" + str(i))
    #
    #
    #     ref_path = "../colorization/Code/Input/ref.jpg"
    #
    #     input_ab = np.zeros((2, 256, 256))
    #     mask = np.zeros((1, 256, 256))
    #
    #     input_ab = np.zeros((2, Xd, Xd))
    #     input_mask = np.zeros((1, Xd, Xd))
    #
    #     (glob_dist_ref, ref_img_fullres) = get_global_histogram(ref_path)
    #     img_pred = cid.net_forward(input_ab, input_mask, glob_dist_ref)
    #     img_pred_withref_fullres = cid.get_img_fullres()
    #
    #     cv2.imwrite("../colorization/Code/Input/ref.jpg", img_pred_withref_fullres)
    #     cv2.imwrite(out_frames_path + 'frame' + str(i) + '.jpg', img_pred_withref_fullres)