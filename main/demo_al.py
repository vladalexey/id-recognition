# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

import pytesseract
import unidecode

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('test_data_path', 'api/uploads', '')
# tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('output_path', 'api/static/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS

config = (" --tessdata-dir 'tessdata' -l vie --oem 1 --psm 7")

def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                    orig = im.copy()
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='O')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                for i, box in enumerate(boxes):
                    reshaped_coords = [box[:8].astype(np.int32).reshape((-1, 1, 2))]
                    cv2.polylines(img, reshaped_coords, True, color=(0, 255, 0),
                                  thickness=2)

                    reshaped_coords = np.asarray(reshaped_coords)
                    roi = img[reshaped_coords[0][0][0][1]:reshaped_coords[0][2][0][1], reshaped_coords[0][0][0][0]:reshaped_coords[0][2][0][0]]
                    
                    text = pytesseract.image_to_string(roi, config=config)

                    text = unidecode.unidecode(text)
                    cv2.putText(img, text, (reshaped_coords[0][0][0][0], reshaped_coords[0][0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])

                with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                          "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)

# if __name__ == '__main__':
    # tf.app.run()

# def delete_prev(path):
    
#     for the_file in os.listdir(path):
#         file_path = os.path.join(path, the_file)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path): shutil.rmtree(file_path)
#         except Exception as e:
#             print(e)
#             continue

# app = Flask(__name__)
# app._static_folder = os.path.basename('static')

# UPLOAD_FOLDER = os.path.join('main', 'uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def hello_world():
#     return render_template('home_al.html')

# @app.route('/upload', methods=['POST', 'GET'])
# def upload_file():

#     if request.method == 'POST':
#         file = request.files['image']
#         filename = file.filename

#         # prepare directory for processing
#         delete_prev(app.config['UPLOAD_FOLDER'])
#         f = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#         # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
#         file.save(f)

#         tf.app.run()

#         print('done')
#         processed_file = os.path.join('data/res', filename)

#         # return render_template('home_al.html', processed_file = processed_file)
#         return redirect(url_for('send_file', filename=filename))
#         print('redirected to', url_for('send_file', filename=filename))
#     else:

#         print('No request')
#         return render_template('home_al.html')

# # @app.route('/show/<filename>')
# # def uploaded_file(filename):
# #     filename = 'http://127.0.0.1:5000/upload/' + filename
# #     return render_template('home_al.html')

# @app.route('/uploaded/<filename>')
# def send_file(filename):
#     return send_from_directory('data/res', filename)

# app.run(debug=True)