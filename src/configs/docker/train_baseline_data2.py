exp_id = '/home/user/weights/'
load_model = '../pretrained/fairmot_dla34.pth'
num_epochs = 30
lr_step = '15'
data_cfg = '../src/lib/cfg/custom.json'
color_weight = 1.05
ball_weight = 0.0
num_teams = 152
K = 10

### test options
input_video = '/home/user/data/'
output_root = '/home/user/data/'
ocr = '/home/user/data/'
write_video = False
# frames_limit = 500


# input_video = '/home/user/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington.mp4'
# output_root = '/home/user/data/docker/2021_01_20_Colorado_at_Washington'
# ocr = '/home/user/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json'



detector_config = '/home/user/weights/yolov3_d53_320_273e_jersey_smallres.py'
detector_path = '/home/user/weights/epoch_90.pth'
classifier_path = '/home/user/weights/model-best.pth'

