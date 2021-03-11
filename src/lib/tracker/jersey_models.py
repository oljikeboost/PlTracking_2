from mmdet.apis import init_detector, inference_detector, inference_batch_detector
from torchvision import transforms
import torch
import torch.jit
import torch.nn as nn
import numpy as np
import cv2
import mmcv
from PIL import Image
from torch2trt import torch2trt

class JerseyModel(torch.nn.Module):

    def __init__(self, inter_size=7):
        super(JerseyModel, self).__init__()

        self.inter_size = inter_size
        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden9 = nn.Sequential(
            nn.Linear(192 * self.inter_size * self.inter_size, 3072),
            nn.ReLU()
        )
        self._hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._digit_length = nn.Sequential(nn.Linear(3072, 4))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))

    # @torch.jit.script_method
    def forward(self, x):
        # print(x.size())
        x = self._hidden1(x)
        x = self._hidden2(x)
        x = self._hidden3(x)
        x = self._hidden4(x)
        x = self._hidden5(x)
        x = self._hidden6(x)
        x = self._hidden7(x)
        x = self._hidden8(x)
        # print(x.size())
        x = x.view(x.size(0), 192 * self.inter_size * self.inter_size)
        x = self._hidden9(x)
        x = self._hidden10(x)

        length_logits = self._digit_length(x)
        digit1_logits = self._digit1(x)
        digit2_logits = self._digit2(x)

        return length_logits, digit1_logits, digit2_logits


    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        # step = int(path_to_checkpoint_file.split('/')[-1][6:-4])


class JerseyDetector():
    def __init__(self,):
        config_file = '/home/ubuntu/oljike/BallTracking/mmdetection/configs/yolo_jersey/yolov3_d53_320_273e_jersey.py'
        checkpoint_file = '/home/ubuntu/oljike/BallTracking/mmdetection/work_dirs/jersey_region_yolov3-320_fullData/epoch_150.pth'
        # build the model from a config file and a checkpoint file

        self.class_model = JerseyModel(7)
        self.class_model.restore('/home/ubuntu/oljike/ocr_jersey/SVHNClassifier-PyTorch/work_dirs/basic_randaug_fulldata/model-best.pth')
        self.class_model.eval().cuda()
        self.ax = 1

        # print("Converting to TRT...")
        # x = torch.rand((1, 3, 54, 54)).cuda()
        # self.class_model = torch2trt(self.class_model, [x], use_onnx=True, max_batch_size=1)
        # self.ax = 2
        # print("Converted to TRT!")

        self.det_model = init_detector(config_file, checkpoint_file, device='cuda')
        self.offset = 2

        self.transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def classify_jersey(self, numpy_image):

        with torch.no_grad():

            numpy_image = cv2.resize(numpy_image, (64, 64))

            image = Image.fromarray(numpy_image)
            image = self.transform(image)
            images = image.unsqueeze(dim=0).cuda()

            length_logits, digit1_logits, digit2_logits = self.class_model(images)

            digit1_prediction = digit1_logits.max(self.ax)[1]
            digit2_prediction = digit2_logits.max(self.ax)[1]

        return [digit1_prediction.item(), digit2_prediction.item()]


    def classify_batch(self, crops):

        with torch.no_grad():
            crops = torch.cat(crops, 0).cuda()
            length_logits, digit1_logits, digit2_logits = self.class_model(crops)

            digit1_prediction = digit1_logits.max(2)[1]
            digit2_prediction = digit2_logits.max(2)[1]

        return [digit1_prediction.tolist(), digit2_prediction.tolist()]

    def infer_batch(self, inp_data):

        all_results = inference_batch_detector(self.det_model, inp_data)
        lost_ids = []
        all_crops = []
        for idx, result in enumerate(all_results):

            if len(result[0]) == 0:
                lost_ids.append(idx)
                continue

            img = inp_data[idx]

            max_res = max(result[0], key=lambda x: x[-1])
            max_prob = max_res[-1]
            if max_prob < 0.6:
                lost_ids.append(idx)
                continue

            max_res = max_res.astype(np.int)
            jersey_crop = img[max_res[1] - self.offset: max_res[3] + self.offset,
                          max_res[0] - self.offset: max_res[2] + self.offset, :]

            if 0 in jersey_crop.shape:
                jersey_crop = img[max_res[1]: max_res[3],
                              max_res[0]: max_res[2], :]

            if 0 in jersey_crop.shape:
                lost_ids.append(idx)
                continue

            jersey_crop = cv2.resize(jersey_crop, (64, 64))
            jersey_crop = Image.fromarray(jersey_crop)
            jersey_crop = self.transform(jersey_crop)
            jersey_crop = jersey_crop.unsqueeze(dim=0)

            all_crops.append(jersey_crop)

        if len(all_crops)==0:
            return [None for _ in range(len(inp_data))]

        jersey_res = self.classify_batch(all_crops)
        output = []
        for l1, l2 in zip(jersey_res[0], jersey_res[1]):
            output.append(''.join([str(x) for x in [l1[0], l2[0]] if x != 10]))

        lost_ids = sorted(lost_ids, reverse=True)
        for i in lost_ids:
            output.insert(i, None)

        return output

    def infer(self, inp_data):
        output = []

        all_results = inference_batch_detector(self.det_model, inp_data)
        for idx, result in enumerate(all_results):

            if len(result[0]) == 0:
                output.append(None)
                continue

            img = inp_data[idx]
            ### In this part we choose the most appropriate detection
            ### The new plan will be to cut all detecitons by some high threshold
            ### And from the remaining ones, choose the one which has the highest 'centerness'.
            ### it is based on assumption that is two good detection overalp each other,
            ### the most centered detection in the right one.
            '''
            remain_inds = result[0][:, 4] > 0.6
            good_results = result[0][remain_inds]        
            '''

            max_res = max(result[0], key=lambda x: x[-1])
            max_prob = max_res[-1]
            if max_prob < 0.6:
                output.append(None)
                continue

            max_res = max_res.astype(np.int)
            jersey_crop = img[max_res[1] - self.offset: max_res[3] + self.offset,
                          max_res[0] - self.offset: max_res[2] + self.offset, :]

            if 0 in jersey_crop.shape:
                jersey_crop = img[max_res[1]: max_res[3],
                              max_res[0]: max_res[2], :]

            if 0 in jersey_crop.shape:
                output.append(None)
                continue

            jersey_res = self.classify_jersey(jersey_crop)
            jersey_res = ''.join([str(x) for x in jersey_res if x != 10])
            output.append(jersey_res)

            # vis_img = inp_data[idx].copy()
            # vis_img = mmcv.imshow_bboxes(vis_img, np.array([max_res]), show=False)
            # vis_img = cv2.putText(vis_img, str(max_prob), (max_res[0], max_res[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
            # cv2.imwrite('crops/crop_{}_{}.jpg'.format(idx, jersey_res), vis_img)

        return output
