exp_id = 'custom_5vals_colors_all_30ep_data2_50ep'
load_model = '../pretrained/fairmot_dla34.pth'
num_epochs = 30
lr_step = '15'
data_cfg = '../src/lib/cfg/custom.json'
color_weight = 1.05
ball_weight = 0.0
num_teams = 152
k = 10

### test options
input_video = '/home/user/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington.mp4'
output_root = '/home/user/data/docker/2021_01_20_Colorado_at_Washington'
ocr = '/home/user/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json'
frames_limit = 10000

