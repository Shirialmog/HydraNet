import os
import time
from shutil import copy
#from SegmentationModule.Logger import LoggerHandler


class SegSettings(object):
    """
    this class is to define the Hydra project settings
    """
    def __init__(self, settings_dict, write_logger):
        self.data_dir = settings_dict['dataset_settings']['data_dir']

        self.load_masks = True
        self.exp_type=settings_dict['exp_type']
        self.percentage=settings_dict['percentage']
        # pre processing settings
        self.pre_process = settings_dict['pre_processing_settings']['pre_process']
        self.apply_transformations = settings_dict['pre_processing_settings']['apply_transformations']
        self.augmentation=settings_dict['pre_processing_settings']['augmentation']
        if self.pre_process:
            self.min_clip_val = settings_dict['pre_processing_settings']['min_val']
            self.max_clip_val = settings_dict['pre_processing_settings']['max_val']
        else:
            self.min_clip_val = None
            self.max_clip_val = None

        # compilation settings
        self.optimizer = settings_dict['compilation_settings']['optimizer']
        self.gamma_decay = settings_dict['compilation_settings']['gamma_decay']
        self.loss = settings_dict['compilation_settings']['loss']
        self.loss_weights = settings_dict['compilation_settings']['loss_weights']
        self.lr_decay_step_size = settings_dict['compilation_settings']['lr_decay_step_size']
        self.lr_decay_policy = settings_dict['compilation_settings']['lr_decay_policy']
        self.initial_learning_rate = settings_dict['compilation_settings']['initial_learning_rate']
        self.weight_decaay = settings_dict['compilation_settings']['weight_decay']
        if self.optimizer == 'adam':
            self.beta_1 = settings_dict['compilation_settings']['beta_1']
            self.beta_2 = settings_dict['compilation_settings']['beta_2']

        # output_settings
        self.simulation_folder = settings_dict['output_settings']['simulation_folder']
        self.model_dir = os.path.join(self.simulation_folder, 'model')
        self.snapshot_dir = os.path.join(self.simulation_folder, 'snapshot')
        self.validation_snapshot_dir = os.path.join(self.simulation_folder, 'validation_snapshot')
        self.weights_dir=os.path.join(self.simulation_folder, 'weights')
        self.test_dir=os.path.join(self.simulation_folder, 'test')
        self.dice_dir = os.path.join(self.simulation_folder, 'dice')
        if not os.path.exists(self.simulation_folder):
            os.mkdir(self.simulation_folder)
        if not os.path.exists(self.snapshot_dir):
            os.mkdir(self.snapshot_dir)
        if not os.path.exists(self.validation_snapshot_dir):
            os.mkdir(self.validation_snapshot_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        if not os.path.exists(self.dice_dir):
            os.mkdir(self.dice_dir)

        # architecture settings
        self.use_skip=settings_dict['architecture_settings']['use_skip']
        self.encoder_name = settings_dict['architecture_settings']['encoder_name']
        self.encoder_depth = settings_dict['architecture_settings']['encoder_depth']
        self.encoder_weights = settings_dict['architecture_settings']['encoder_weights']
        self.weights_init = settings_dict['compilation_settings']['weights_init']
        self.decoder_use_batchnorm = settings_dict['architecture_settings']['decoder_use_batchnorm']
        self.decoder_channels = settings_dict['architecture_settings']['decoder_channels']
        self.in_channels = settings_dict['architecture_settings']['in_channels']
        self.classes = settings_dict['architecture_settings']['classes']
        self.dimension = settings_dict['architecture_settings']['dimension']
        self.input_size = settings_dict['architecture_settings']['input_size']
        if self.loss == 'CrossEntropyLoss':
            self.activation = 'identity'
        else:
            self.activation = settings_dict['architecture_settings']['activation']

        # training settings
        self.train_model = settings_dict['training_settings']['train_model']
        self.batch_size = settings_dict['training_settings']['batch_size']
        self.num_epochs = settings_dict['training_settings']['num_epochs']
        self.copy_code()


    # copy experiment code
    def copy_code(self):
        file_path=r'C:\Users\user_project\PycharmProjects\deep_learning_project\ayelet_shiri\SegmentationModule\transfer.py'
        target_path=self.simulation_folder+'\code.txt'
        copy(file_path, target_path)

