#Thirst Library
import multiprocessing as mp
import cv2
import os
import concurrent.futures
import yaml
import numpy as np
import math
from paddle.inference import Config as PPConfig
from paddle.inference import create_predictor
import time


from deploy.utils import Config, subprocess_ocr_request, submit_ocr_request, group_element, REVERSE_MAP
# from utils import Config, subprocess_ocr_request, submit_ocr_request, group_element, REVERSE_MAP


class DetectorSJ(object):
    def __init__(self):
        self.cfg:Config  = Config() 
        self.model = PaddleDectector(
            self.cfg.store_path,
            self.cfg.cfg_path,
            batch_size = self.cfg.batchsize,
            device=self.cfg.device,
            cpu_threads=self.cfg.n_processes,
            threshold=self.cfg.threshold)

    # predict from image
    def ready(self):
        return str(type(self.model))
   
    # @profile
    def infer(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        inputs = [np.array(i) for i in inputs]
        detection_results = self.model.predict_image(inputs)

        #OCR            
        # parent_connections = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for box in detection_results:
        #         parent_conn, child_conn = mp.Pipe()
        #         parent_connections += [parent_conn]      
        #         executor.submit(subprocess_ocr_request, (box['page_id'], box['xyxy'][1]), REVERSE_MAP[box['cls']],
        #                     inputs[box['page_id']][box['xyxy'][1]: box['xyxy'][3], box['xyxy'][0]: box['xyxy'][2]],
        #                     self.cfg.matpix_id, self.cfg.matpix_key, self.cfg.s3_bucket_name,
        #                     child_conn)
        # elements = []
        # for parent_connection in parent_connections:
        #     elements += [parent_connection.recv()]
        elements = submit_ocr_request(detection_results=[[(box['page_id'], box['xyxy'][1]), REVERSE_MAP[box['cls']],
                                                inputs[box['page_id']][box['xyxy'][1]: box['xyxy'][3], box['xyxy'][0]: box['xyxy'][2]] ]
                                                for box in detection_results],
                                      config=self.cfg)

        #Postprocesing
        elements = sorted(elements, key=lambda x: x["idx"])
        questions = group_element(elements)

        return questions     

#########################################
### 
##       Paddle module
###
#########################################
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Source : https://github.com/PaddlePaddle/PaddleDetection/tree/develop/deploy/python
"""

def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR 
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info




def preprocess(im, preprocess_ops):
    # process image by preprocess_ops
    im_info = {
        'scale_factor': np.array(
            [1., 1.], dtype=np.float32),
        'im_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info

###########################################################################################################3

# DETECTOR


##########################################################################################################
       
class PaddleDectector(object):
    #https://github.com/PaddlePaddle/PaddleDetection/tree/develop/deploy/python/infer.py

    def __init__(self,
                 model_dir,
                 cfg_path,
                 batch_size=1,
                 device='CPU',
                 cpu_threads=2,
                 threshold=0.5,):
        self.pred_config = PredictConfig(cfg_path)
        self.predictor, self.config = load_predictor( model_dir,run_mode=self.pred_config.mode, device=device, cpu_threads=cpu_threads)
        self.batch_size = batch_size
        self.threshold = threshold

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        input_im_lst = []
        input_im_info_lst = []
        for pil_img in image_list:
            pil_img = np.array(pil_img)
            im, im_info = preprocess(pil_img, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)

        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def predict(self):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes_num, np_boxes = np.array([0]), None

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        if len(output_names) == 1:
            # some exported model can not get tensor 'bbox_num' 
            np_boxes_num = np.array([len(np_boxes)])
        else:
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
        result = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results


    def predict_image(self, image_list):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            inputs = self.preprocess(batch_image_list)
            # model prediction
            result = self.predict()
            # postprocess
            result = {k: v for k, v in result.items() if v is not None} #self.postprocess(inputs, result)
            results.append(result)
        results = self.merge_batch_result(results)

        bbox_results = []
        idx = 0
        for i, box_num in enumerate(results['boxes_num']):
            img_id = i
            if 'boxes' in results:
                boxes = results['boxes'][idx:idx + box_num] 
                boxes = boxes[ boxes[:,1] > self.threshold].tolist()
                bbox_results += [{
                    'page_id': img_id,
                    'cls': int(box[0]),
                    'xyxy': [int(box[2]), int(box[3]), int(box[4]+1), int(box[5]+1)]  # xyxy 
                    } for box in boxes]
            idx += box_num
        return bbox_results


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, cfg_path):
        # parsing Yaml config for Preprocess
        with open(cfg_path) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.mode = yml_conf['mode']
        self.preprocess_infos = yml_conf['Preprocess']
        self.labels = yml_conf['label_list']


def load_predictor(model_dir,
                   run_mode='paddle',
                   device='CPU',
                   cpu_threads=2, **kwargs):
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError("Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(run_mode, device))
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    config = PPConfig(infer_model, infer_params)
    if device == 'GPU':
        config.enable_use_gpu(200, 0)         # initial GPU memory(M), device ID
        config.switch_ir_optim(True)         # optimize graph and fuse op
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)

    predictor = create_predictor(config)
    return predictor, config