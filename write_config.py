import configparser
import yaml

dsn_config = {

    # Sizes
    'feature_dim': 64,  # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters': 10,
    'num_seeds': 200,  # Used for MeanShift, but not BlurringMeanShift
    'epsilon': 0.05,  # Connected Components parameter
    'sigma': 0.02,  # Gaussian bandwidth parameter
    'subsample_factor': 5,
    'min_pixels_thresh': 500,

    # Differentiable backtracing params
    'tau': 15.,
    'M_threshold': 0.3,

    # Robustness stuff
    'angle_discretization': 100,
    'discretization_threshold': 0.,

}

uois3d_config = {

    # Padding for RGB Refinement Network
    'padding_percentage': 0.25,

    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology': True,
    'open_close_morphology_ksize': 9,

    # Largest Connected Component for IMP module
    'use_largest_connected_component': True,

    'final_close_morphology': True,
}

rrn_config = {

    # Sizes
    'feature_dim': 64,  # 32 would be normal
    'img_H': 224,
    'img_W': 224,

    # architecture parameters
    'use_coordconv': False,

}
#
# config = configparser.RawConfigParser()
# config.optionxform = lambda option: option
#
# save_path = 'uois.config'
# # config = configparser.ConfigParser()
config = dict()
config['dsn'] = dsn_config
config['rrn'] = rrn_config
config['uois3d'] = uois3d_config

fp = open('config.yaml', 'w')
fp.write(yaml.dump(config))
fp.close()

# with open(save_path, 'w', encoding='utf-8') as file:
#    config.write(file)  # 数据写入配置文件

def load_config(config_path):
    fp = open(config_path, 'r')
    st = fp.read()
    fp.close()
    config = yaml.load(st, Loader=yaml.FullLoader)
    return config

test_config = load_config('config.yaml')
