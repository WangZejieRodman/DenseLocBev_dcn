# Warsaw University of Technology

import os
import configparser
import time
import numpy as np

# =========================================================
# 修改点 1: 导入 BEVQuantizer
# =========================================================
from datasets.quantization import PolarQuantizer, CartesianQuantizer, BEVQuantizer


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)  # 最终全局描述子的维度

        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = params.get('coordinates', 'polar')
        # =========================================================
        # 修改点 2: 在 assert 中允许 'bev'
        # =========================================================
        assert self.coordinates in ['polar', 'cartesian', 'bev'], f'Unsupported coordinates: {self.coordinates}'

        if 'polar' in self.coordinates:
            # 极坐标量化的3个步长: 扇区(度), 环(米), Z轴(米)
            self.quantization_step = tuple([float(e) for e in params['quantization_step'].split(',')])
            assert len(
                self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(quant_step=self.quantization_step)

        elif 'cartesian' in self.coordinates:
            # 笛卡尔坐标系的单一量化步长
            self.quantization_step = params.getfloat('quantization_step')
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)

        # =========================================================
        # 修改点 3: 新增 bev 坐标系的参数解析逻辑
        # =========================================================
        elif 'bev' in self.coordinates:
            # 解析 BEV 特有的参数

            # 1. 解析 coords_range (例如: -10., -10, -4, 10, 10, 8)
            if 'coords_range' in params:
                self.coords_range = [float(e) for e in params['coords_range'].split(',')]
            else:
                # 默认值 (Chilean数据集配置)
                self.coords_range = [-10., -10, -4, 10, 10, 8]

            # 2. 解析 div_n (例如: 256, 256, 32)
            if 'div_n' in params:
                self.div_n = [int(e) for e in params['div_n'].split(',')]
            else:
                # 默认值
                self.div_n = [256, 256, 32]

            # 3. 解析 in_channels (Z轴层数, 通常等于 div_n[2])
            # 这个参数会被 model_factory 用来初始化 Backbone
            self.in_channels = params.getint('in_channels', self.div_n[2])

            # 4. 实例化 BEVQuantizer
            self.quantizer = BEVQuantizer(coords_range=self.coords_range, div_n=self.div_n)

        else:
            raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")

        # 使用余弦相似度还是欧氏距离
        # 当使用欧氏距离时，embedding 归一化是可选的
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)

        # Backbone 输出的局部特征维度 (仅用于基于 MinkNet 的模型)
        self.feature_size = params.getint('feature_size', 256)

        # 以下参数主要用于原本的 MinkFPN/ResNet 结构
        # 对于 MinkBEVBackbone，这些参数可能不会被用到，但为了兼容性保留解析
        if 'planes' in params:
            self.planes = tuple([int(e) for e in params['planes'].split(',')])
        else:
            self.planes = tuple([32, 64, 64])

        if 'layers' in params:
            self.layers = tuple([int(e) for e in params['layers'].split(',')])
        else:
            self.layers = tuple([1, 1, 1])

        self.num_top_down = params.getint('num_top_down', 1)
        self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)
        self.block = params.get('block', 'BasicBlock')
        self.pooling = params.get('pooling', 'GeM')

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            # =========================================================
            # 修改点 4: 打印新增的 BEV 参数
            # =========================================================
            elif e == 'coords_range':
                print(f'coords_range: {param_dict[e]}')
            elif e == 'div_n':
                print(f'div_n: {param_dict[e]}')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class TrainingParams:
    """
    模型训练参数
    """

    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(
            model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)  # 模型保存频率 (epochs)
        self.num_workers = params.getint('num_workers', 0)

        # 全局描述子的初始 batch size
        self.batch_size = params.getint('batch_size', 64)
        # 当 batch_split_size 非零时，启用多阶段反向传播
        self.batch_split_size = params.getint('batch_split_size', None)

        # 动态 batch size 调整
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                if 'scheduler_milestones' in params:
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    self.scheduler_milestones = [self.epochs + 1]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss').lower()
        if 'contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)  # Margin used in loss function
        elif self.loss == 'truncatedsmoothap':
            self.positives_per_query = params.getint("positives_per_query", 4)
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)  # Margin used in loss function

        # 相似度度量: 余弦相似度 或 欧氏距离
        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)  # 增强模式
        self.set_aug_mode = params.getint('set_aug_mode', 1)  # Set 增强模式
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file', None)

        # 读取模型参数
        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')