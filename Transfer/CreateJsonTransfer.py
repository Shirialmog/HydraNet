import json
import os

experiments_dict = {'transfer':['transfer', 'scratch','imagenet']} ##resnet

def create_json(father_folder_path, task,percentage, exp_start_ind=0):
    exp_ind = exp_start_ind

    ##a for loop over different experiments options should be added here

    for exp_type in experiments_dict['transfer']:

        main_dir = os.path.join(father_folder_path, 'Transfer_exp', task,'truncate_0.2',percentage,'exp_{}'.format(exp_ind))
        if not os.path.exists(main_dir):
            os.mkdir(main_dir)
        exp_dir=os.path.join(main_dir,exp_type)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        json_data = {}

        json_data['exp_type']=exp_type
        json_data['percentage']=percentage
        # dataset settings
        json_data['dataset_settings'] = {'definition_file_dir': father_folder_path,
                                       'data_dir': father_folder_path +'/'+task+'/'+percentage}

        # pre processing settings
        json_data['pre_processing_settings'] = {'pre_process': True, #[True, False],
                                                'min_val': -300,
                                                'max_val': 600,
                                                'apply_transformations': False,
                                                'augmentation': False
                                                }

        # compilation settings
        json_data['compilation_settings'] = {'loss': 'CrossEntropyLoss',
                                             'loss_weights': {'background': 1, 'organ': 10},
                                             'weights_init': 'imagenet',
                                             'initial_learning_rate': 0.00001,
                                             'gamma_decay': 0.5,
                                             'lr_decay_policy': 'step',
                                             'lr_decay_step_size': 5000,
                                             'optimizer': 'adam',
                                             'weight_decay': 0.0001,
                                             'beta_1': 0.5,
                                             'beta_2': 0.999}

        # output_settings
        json_data['output_settings'] = {
            'simulation_folder': exp_dir}

        # architecture settings
        if exp_type == 'imagenet':
            encoder_weights = 'imagenet'
        else:
            encoder_weights = None

        json_data['architecture_settings'] = {'encoder_name': 'densenet121',
                                              'encoder_depth': 5,
                                              'encoder_weights': encoder_weights,
                                              'decoder_use_batchnorm': True,
                                              'decoder_channels': [256, 128, 64, 32, 16],
                                              'in_channels': 3,
                                              'classes': 2,
                                              'dimension': 2,
                                              'activation': 'softmax',
                                              'input_size': (3, 384, 384),
                                              'use_skip': None}

        # training settings
        json_data['training_settings'] = {'train_model': True,
                                          'batch_size': 4,
                                          'num_epochs': 10}


        file_path = os.path.join(exp_dir, 'exp_{}.json'.format(exp_ind))
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)


if __name__== '__main__':
    user='remote'
    folder_path= r'G:\Deep learning\Datasets_organized\small_dataset'
    exp_start_ind = 3
    task='Uterus'
    percentage='100'
    create_json(folder_path, task,percentage, exp_start_ind)

