Hello World! THIS IS REPOSITORY FOR PLAYER TRACKING!

Before running install Docker 

To run a raw video file for tracking, run the following:
1. `git clone https://github.com/oljikeboost/Tracking`
2. `cd Tracking`
3. `nvidia-docker build -t tracking .` (here we build the docker image from which we create container)
4. `nvidia-docker run -v path/to/directory/with/video/and/ocr:/home/user/data/GAME_NAME --name inference -t -d tracking configs/docker/train_baseline_data2.py`
5. The directory from step 4 must contain the video and the ocr. The output of the container will be in this directory.
6. To check the process of the container, run `docker log inference -f`