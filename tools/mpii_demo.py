
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from pose.utils.osutils import mkdir_p, isfile, isdir, join
import pose.models as models
from scipy.ndimage import gaussian_filter, maximum_filter
import cv2
import numpy as np

def load_image(imgfile, w, h ):
    image = cv2.imread(imgfile)
    image = cv2.resize(image, (w, h))
    image = image[:, :, ::-1]  # BGR -> RGB
    image = image / 255.0
    image = image - np.array([[[0.4404, 0.4440, 0.4327]]])  # Extract mean RGB
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = image[np.newaxis, :, :, :]
    return image

def load_model(arch='hg', stacks=2, blocks=1, num_classes=16, mobile=True,
               resume='checkpoint/pytorch-pose/mpii_hg_s2_b1_mobile/checkpoint.pth.tar'):
    # create model
    model = models.__dict__[arch](num_stacks=stacks, num_blocks=blocks, num_classes=num_classes, mobile=mobile)
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    model.eval()
    return model

def inference(model, image):
    model.eval()
    input_tensor = torch.from_numpy(image).float().cuda()
    output = model(input_tensor)
    output = output[-1]
    output = output.data.cpu()
    print(output.shape)
    kps = post_process_heatmap(output[0,:,:,:])
    return kps


def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[0]):
        _map = heatMap[i, :, :]
        _map = gaussian_filter(_map, sigma=1)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))

    kp = np.array(kplst)
    return kp


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

def render_kps(cvmat, kps, scale_x, scale_y):
    for _kp in kps:
        _x, _y, _conf = _kp
        if _conf > 0.2:
            cv2.circle(cvmat, center=(int(_x*4*scale_x), int(_y*4*scale_y)), color=(0,0,255), radius=5)

    return cvmat


def main():
    model = load_model()
    in_res_h , in_res_w = 192, 192

    imgfile = "/home/yli150/sample.jpg"
    image = load_image(imgfile, in_res_w, in_res_h)
    print(image.shape)

    kps = inference(model, image)

    cvmat = cv2.imread(imgfile)
    scale_x = cvmat.shape[1]*1.0/in_res_w
    scale_y = cvmat.shape[0]*1.0/in_res_h
    render_kps(cvmat, kps, scale_x, scale_y)
    print(kps)
    cv2.imshow('x', cvmat)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()