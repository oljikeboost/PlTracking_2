exp_id = 'baseline_data2_r50'
# load_model = '../pretrained/fairmot_dla34.pth'
num_epochs = 30
lr_step = '15'
data_cfg = '../src/lib/cfg/custom.json'
color_weight = 1.05
ball_weight = 0.0
num_teams = 152
arch = 'dla_60'

### test options
# input_video = '/home/ubuntu/oljike/data/videos2/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington.mp4'
# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'
# ocr = '/home/ubuntu/oljike/data/videos2/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json'

# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'
# input_video = '/home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_orig.mp4'
# ocr = '/home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_ocr.json'


output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'
input_video = '/home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_orig.mp4'
ocr = '/home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_ocr.json'
