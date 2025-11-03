# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
from PIL import Image
import numpy as np
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
import numpy as np
from PIL import Image
from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


import logging
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_gpu_ram():
    gpu_ram_info = []
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            used_memory = torch.cuda.memory_allocated(i) / (1024 ** 2)
            free_memory = (total_memory - used_memory)
            
            # gpu_ram_info.append({
            #     "GPU": i,
            #     "Total Memory (MB)": total_memory,
            #     "Used Memory (MB)": used_memory,
            #     "Free Memory (MB)": free_memory
            # })
            return free_memory
    else:
        print("CUDA is not available on this system.")
    
    return 



def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')

    parser.add_argument(
        '--sim-test', action='store_true', help='Use test similar to DocTamper paper')
    parser.add_argument(
        '--nm', type=str, help='Use test similar to DocTamper paper')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    if args.sim_test:
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        simp_test(model, data_loader, name_model=args.nm)

    else:
        if not distributed:
            # warnings.warn(
            #     'SyncBN is only supported with DDP. To be compatible with DP, '
            #     'we convert SyncBN to BN. Please use dist_train.sh which can '
            #     'avoid this error.')
            if not torch.cuda.is_available():
                assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                    'Please use MMCV >= 1.4.4 for CPU training!'
            model = revert_sync_batchnorm(model)
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            results = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                False,
                args.opacity,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)
            results = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
                False,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                warnings.warn(
                    'The behavior of ``args.out`` has been changed since MMSeg '
                    'v0.16, the pickled outputs could be seg map as type of '
                    'np.array, pre-eval results or file paths for '
                    '``dataset.format_results()``.')
                print(f'\nwriting results to {args.out}')
                mmcv.dump(results, args.out)
            if args.eval:
                eval_kwargs.update(metric=args.eval)
                metric = dataset.evaluate(results, **eval_kwargs)
                metric_dict = dict(config=args.config, metric=metric)
                mmcv.dump(metric_dict, json_file, indent=4)
                if tmpdir is not None and eval_on_format_results:
                    # remove tmp dir when cityscapes evaluation
                    shutil.rmtree(tmpdir)

class IOUMetric:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes) #remove -1 value or sth else
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask], 
            minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, prediction, gt):
        self.hist += self._fast_hist(prediction.flatten(), gt.flatten())
    def evaluate(self):
        cm = self.hist / self.hist.sum()
        print("\n===============histogram===========\n", cm[0], "\n",cm[1])
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

import shutil


def simp_test(model, dataloader, name_model=""):
    # print(count_parameters(model));exit()
    from tqdm import tqdm
    model.eval()
    iou=IOUMetric(2)
    precisons, recalls, fps, fns = [], [], [], []
    save_path=f"/home/k64t/Tampereddoc/zda_reimplementation/visual/segment-{name_model}"
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(dataloader)):
            # if batch_idx > 30: break
            path = batch_samples['img_metas'][0].data[0][0]
            path = path.get("filename")
            dataset_name = os.path.basename(os.path.dirname(path))
            image_name = os.path.basename(path)
            
            pred = model(return_loss=False, **batch_samples)
            pred = pred[0]
            img_metas = batch_samples['img_metas'][0].data[0]
            target = np.array(Image.open(img_metas[0]['filename'].replace("images", "annotation").replace("jpg", "png")))
            if batch_idx < 30:
                os.makedirs(os.path.join(save_path, dataset_name), exist_ok = True)
                array = (pred+0) * 255
                image = Image.fromarray(array.astype(np.uint8))
                # Save image
                image.save(os.path.join(save_path, dataset_name, image_name.replace(".jpg", '.png')))
                shutil.copy(path, os.path.join(save_path, dataset_name, image_name))
                array = (target+0) * 255
                imageT = Image.fromarray(array.astype(np.uint8))
                imageT.save(os.path.join(save_path, dataset_name, image_name.replace(".jpg", 'Target.png')))
            
            predt = pred   + 0
            targt = target + 0
            matched = (predt*targt).sum((0,1))
            pred_sum = predt.sum((0,1))
            target_sum = targt.sum((0,1))
            precisons.append((matched/(pred_sum+1e-8)).mean().item())
            recalls.append((matched/(target_sum+1e-8)).mean().item())
            #check
            false_positives = (predt * (1 - targt)).sum((0, 1))
            false_negatives = ((1 - predt) * targt).sum((0, 1))
            fps.append(false_positives.mean().item())
            fns.append(false_negatives.mean().item())
            iou.add_batch(pred,target)
        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        fps = np.array(fps).mean()
        fns = np.array(fns).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{} fps:{}, fns:{}'.format(iu,precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8)), fps, fns))
    return iu, precisons, recalls, fps, fns ,acc

if __name__ == '__main__':
    import time
    while check_gpu_ram() < 5: 
        print("\t\t--->!!! RAM only has:", check_gpu_ram())
        time.sleep(120)
    main()
