# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import glob
import json
import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from ..customized_pipe import tamaugmentation

@DATASETS.register_module()
class DocTamper(CustomDataset):
    """DocTamper dataset. https://github.com/qcf-568/DocTamper.

    In segmentation map annotation for DocTamper, 0 stands for background, which
    is not included in 1 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ("normal",'tampered')
    PALETTE = [[0, 0, 0], [120, 120, 120]]
    
    # CLASSES = ('tampered')
    # PALETTE = [[120, 120, 120]]

    

    def __init__(self, **kwargs):
        super(DocTamper, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        self.tamper_augment = kwargs.get('use_tamaug', False)
        print("\t\tself,tamper_aug---->", self.tamper_augment)
        if self.tamper_augment:
            _aug_root = osp.join(kwargs.get('data_root'), "images/synthetic/annotation")
            self.augmented_annotation = {}
            for file_path in glob.glob(osp.join(_aug_root, "*.json"))    :
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    imgID =  osp.basename(data['org'])
                    if len(data["tampered_part"]) > 2:
                        self.augmented_annotation[imgID] = data["tampered_part"]
            print("\n\t\t\t====Load tampered augemtation with ==>", len(self.augmented_annotation), " file!\n\n")

        
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        if self.tamper_augment:
            results['tampered_parts'] = self.augmented_annotation.get(img_info['filename'], [])
        results = self.pipeline(results)
        return results

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
