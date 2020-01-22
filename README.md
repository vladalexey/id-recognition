** Build an ID Card recognition API with implementing different image processing methods on top of Text-Detection-CTPN (below) to better tackle text detection in old, damaged and artifacts-deposited ID cards **

# text-detection-ctpn

Scene text detection based on ctpn (connectionist text proposal network). It is implemented in tensorflow. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/).

# setup
nms and bbox utils are written in cython, hence you have to build the library first.

cd utils/bbox
chmod +x make.sh
./make.sh
It will generate a nms.so and a bbox.so in current folder.

# demo
follow setup to build the library
download the ckpt file from google drive or baidu yun
put checkpoints_mlt/ in text-detection-ctpn/
put your images in data/demo, the results will be saved in data/res, and run demo in the root
python3 ./main/demo_al.py
