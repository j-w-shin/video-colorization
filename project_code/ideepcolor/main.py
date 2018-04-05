import numpy as np
import cv2
import caffe
from data import colorize_image as CI
import glob
import os
from skimage import io, color
from subprocess import call


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


def get_global_histogram(gt_glob_net, ref_path):
    ref_img_fullres = caffe.io.load_image(ref_path)
    img_glob_dist = (255 * caffe.io.resize_image(ref_img_fullres, (Xd, Xd))).astype('uint8')  # load image
    gt_glob_net.blobs['img_bgr'].data[...] = img_glob_dist[:, :, ::-1].transpose((2, 0, 1))  # put into
    gt_glob_net.forward()
    glob_dist_in = gt_glob_net.blobs['gt_glob_ab_313_drop'].data[0, :-1, 0, 0].copy()
    return glob_dist_in, ref_img_fullres


def vanilla_network_colorize(prototxt_path, caffemodel_path, bw_frames_path, out_frames_path):
    img = cv2.imread(bw_frames_path + "frame0.jpg")
    height, width, _ = img.shape
    # video = VideoWriter('a.avi', frameSize=(width, height))
    # video.open()
    # video = cv2.VideoWriter('a.mp4', -1, fps, (width, height))

    gpu_id = -1

    colorModel = CI.ColorizeImageCaffe(Xd)

    colorModel.prep_net(gpu_id, prototxt_path, caffemodel_path)

    mask = np.zeros((1, Xd, Xd))
    input_ab = np.zeros((2, Xd, Xd))

    for i in range(len(glob.glob(bw_frames_path + "*"))):
        print(i)
        colorModel.load_image(bw_frames_path + "frame" + str(i) + ".jpg")

        img_out = colorModel.net_forward(input_ab, mask)
        img_out_fullres = colorModel.get_img_fullres()

        # cv2.imwrite(temp_frames_path + "frame" + str(i) + ".jpg", img_out_fullres)
        # frame = cv2.imread(temp_frames_path + "frame" + str(i) + ".jpg")
        # video.write(img_out_fullres)
        cv2.imwrite(out_frames_path + 'frame' + str(i) + '.jpg', img_out_fullres)

        # print(img_out_fullres.shape)
        # exit(1)


    cv2.destroyAllWindows()
    video.release()


def global_hist_network_colorize(prototxt_path, caffemodel_path, global_prototxt_path, global_caffemodel_path, bw_frames_path, out_frames_path):
    img = cv2.imread(bw_frames_path + "frame0.jpg")
    height, width, _ = img.shape
    # video = cv2.VideoWriter('a.mp4', -1, fps, (width, height))

    gpu_id = -1

    cid = CI.ColorizeImageCaffeGlobDist(Xd)
    cid.prep_net(gpu_id, prototxt_path, caffemodel_path)

    # Global distribution network - extracts global color statistics from an image
    gt_glob_net = caffe.Net(global_prototxt_path, global_caffemodel_path, caffe.TEST)

    ref_path = 'ref.jpg'

    input_ab = np.zeros((2, Xd, Xd))
    input_mask = np.zeros((1, Xd, Xd))

    for i in range(len(glob.glob(bw_frames_path + "*"))):
        print(i)

        img_path = bw_frames_path + 'frame' + str(i) + '.jpg'

        if i == 0:
            cid.load_image(img_path)
            img_pred = cid.net_forward(input_ab, input_mask)
            img_pred_auto_fullres = cid.get_img_fullres()
            cv2.imwrite("ref.jpg", img_pred_auto_fullres)

            cv2.imwrite(out_frames_path + 'frame' + str(i) + '.jpg', img_pred_auto_fullres)
            # video.write(img_pred_auto_fullres)
            continue


        cid.load_image(img_path)

        (glob_dist_ref, ref_img_fullres) = get_global_histogram(gt_glob_net, ref_path)
        img_pred = cid.net_forward(input_ab, input_mask, glob_dist_ref)
        img_pred_withref_fullres = cid.get_img_fullres()

        cv2.imwrite(out_frames_path + 'frame' + str(i) + '.jpg', img_pred_withref_fullres)
        # video.write(img_pred_withref_fullres)
        cv2.imwrite(ref_path, img_pred_withref_fullres)

    cv2.destroyAllWindows()
    # video.release()


def feature_map_network_colorize(prototxt_path, caffemodel_path, global_prototxt_path, global_caffemodel_path, bw_frames_path, out_frames_path):

    # Choose gpu to run the model on
    gpu_id = -1

    # Initialize colorization class
    cid = CI.ColorizeImageCaffe(Xd=256)

    # Load the model
    cid.prep_net(gpu_id, prototxt_path, caffemodel_path)

    # Load the image
    img_path = bw_frames_path + 'frame0.jpg'
    cid.load_image(bw_frames_path + 'frame0.jpg') # load an image

    mask = np.zeros((1,256,256)) # giving no user points, so mask is all 0's
    input_ab = np.zeros((2,256,256)) # ab values of user points, default to 0 for no input
    img_pred = cid.net_forward(input_ab,mask) # run model, returns 256x256 image
    img_pred_auto_fullres = cid.get_img_fullres()
    cv2.imwrite("../colorization/Code/Input/ref.jpg", img_pred_auto_fullres)
    cv2.imwrite(out_frames_path + 'frame0.jpg', img_pred_withref_fullres)

    engine = matlab.engine.start_matlab()
    engine.Run_All('frame0', 'ref', nargout=0)

    # call(['mv', '../colorization/Code/Result/frame0_pixmap_sat.png', ])

    rgb = io.imread('../colorization/Code/Result/frame0_pixmap_sat.png')
    lab = color.rgb2lab(rgb)

    pix_idxs    = []
    pix_ab_vals = []

    for row in xrange(len(lab)):
        for col in xrange(len(lab[0])):
            if list(lab[row][col]) != [0, 0, 0]:
                pix_idxs.append((row, col))
                pix_ab_vals.append(lab[row][col])

    input_ab = np.zeros((2,256,256))
    mask = np.zeros((1,256,256))

    for i in xrange(len(pix_idxs)):
        print i + 1, "..."
        # add a blue point in the middle of the image
        (input_ab,mask) = put_point(input_ab,mask,[pix_idxs[i][0],pix_idxs[i][1]],3,[pix_ab_vals[i][1],pix_ab_vals[i][2]])

        # call forward
        img_out = cid.net_forward(input_ab,mask)

    cv2.imwrite("../colorization/Code/Input/ref.jpg", img_pred_auto_fullres)
    cv2.imwrite(out_frames_path + 'frame.jpg', img_pred_withref_fullres)



def put_point(input_ab,mask,loc,p,val):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       half-patch size
    # val         2 tuple      (a,b) value of user input
    input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
    mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
    return (input_ab,mask)



if __name__ == '__main__':
    color_frames_path = "color_frames/"
    bw_frames_path = "bw_frames/"
    out_frames_path = "out_frames_vanilla/"
    video_path = "YUP++/camera_moving/RushingRiver/RushingRiver_moving_cam_31.mp4"

    # vanilla mode
    # prototxt_path = 'ideepcolor/models/reference_model/deploy_nodist.prototxt'
    # caffemodel_path = 'ideepcolor/models/reference_model/model.caffemodel'

    # global histogram mode
    prototxt_path = os.getcwd() + "/models/global_model/deploy_nodist.prototxt"
    caffemodel_path = os.getcwd() + "/models/global_model/global_model.caffemodel"
    global_prototxt_path = os.getcwd() + "/models/global_model/global_stats.prototxt"
    global_caffemodel_path = os.getcwd() + "/models/global_model/dummy.caffemodel"

    Xd = 256

    fps = video_to_frames(video_path, color_frames_path)
    make_bw_frames(color_frames_path, bw_frames_path)

    vanilla_network_colorize(prototxt_path, caffemodel_path, bw_frames_path, out_frames_path)
    # global_hist_network_colorize(prototxt_path, caffemodel_path, global_prototxt_path, global_caffemodel_path, bw_frames_path, out_frames_path)
