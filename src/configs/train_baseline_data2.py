exp_id = 'custom_5vals_colors_all_30ep_data2_50ep'
load_model = '../pretrained/fairmot_dla34.pth'
num_epochs = 30
lr_step = '15'
data_cfg = '../src/lib/cfg/custom.json'
color_weight = 1.05
ball_weight = 0.0
num_teams = 152

### test options
# input_video = '/home/ubuntu/oljike/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington.mp4,\
#               /home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_orig.mp4,\
#               /home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_orig.mp4'
# ocr = '/home/ubuntu/oljike/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json,\
#       /home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_ocr.json,\
#       /home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_ocr.json'
# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'


# input_video = '/home/ubuntu/oljike/data/demo_vids/2020_12_20_OklahomaState_at_Texas/2020_12_20_OklahomaState_at_Texas.mp4,\
#               /home/ubuntu/oljike/data/demo_vids/2021_01_04_WestVirginia_at_OklahomaState/2021_01_04_WestVirginia_at_OklahomaState.mp4,\
#                 /home/ubuntu/oljike/data/demo_vids/2021_02_08_OklahomaState_at_Kansas/2021_02_08_OklahomaState_at_Kansas.mp4,\
#               /home/ubuntu/oljike/data/demo_vids/2021_02_27_OklahomaState_at_Oklahoma/2021_02_27_OklahomaState_at_Oklahoma.mp4,\
#               /home/ubuntu/oljike/data/demo_vids/2021_03_01_Oklahoma_at_OklahomaState/2021_03_01_Oklahoma_at_OklahomaState.mp4,\
#                 /home/ubuntu/oljike/data/demo_vids/2021_03_11_OklahomaState_at_WestVirginia/2021_03_11_OklahomaState_at_WestVirginia.mp4,\
#               /home/ubuntu/oljike/data/demo_vids/2021_03_12_OklahomaState_at_Baylor/2021_03_12_OklahomaState_at_Baylor.mp4'
#
# ocr = '/home/ubuntu/oljike/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json,\
#       /home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_ocr.json,\
#       /home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_ocr.json'
# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'
#

input_video = '/home/ubuntu/oljike/data/demo_vids/2021_03_12_OklahomaState_at_Baylor/2021_03_12_OklahomaState_at_Baylor.mp4'

ocr = '/home/ubuntu/oljike/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json,\
      /home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_ocr.json,\
      /home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_ocr.json'
output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'





# input_video = '/home/ubuntu/oljike/data/videos1/demo_vids/1.mp4'
# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep/frame_60'
# ocr = '/home/ubuntu/oljike/data/videos1/demo_vids/ocr/1_ocr.json'
# frames_limit = 2000


# input_video = '/home/ubuntu/oljike/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington.mp4'
# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep/frame_60_crct'
# ocr = '/home/ubuntu/oljike/data/demo_vids/2021_01_20_Colorado_at_Washington/2021_01_20_Colorado_at_Washington_ocr.json'
# frames_limit = 10000


# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'
# input_video = '/home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_orig.mp4'
# ocr = '/home/ubuntu/oljike/data/demo_vids/2020_11_28_TexasSouthern_at_OklahomaState/2020_11_28_TexasSouthern_at_OklahomaState_ocr.json'


# output_root = '../demos/custom_5vals_colors_all_30ep_data2_50ep'
# input_video = '/home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_orig.mp4'
# ocr = '/home/ubuntu/oljike/data/demo_vids/2021_03_13_OklahomaState_at_Texas/2021_03_13_OklahomaState_at_Texas_ocr.json'
