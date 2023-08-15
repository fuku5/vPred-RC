import re
import numpy as np

def load_npz_all(npz_dir, model_name, train):
    if train:
        suffix = 'train'
    else:
        suffix = 'test'
    npz_paths = list(npz_dir.glob('{}__*-{}.npz'.format(model_name, suffix)))

    rtn = dict()

    for npz_path in npz_paths:
        dataset_name = re.search(r'__(.*)-{}'.format(suffix), str(npz_path)).groups(0)[0]
        npz = np.load(npz_path)
        rtn[dataset_name] = {
            'known': dataset_name in model_name
        }
        rtn[dataset_name].update(npz)
    return rtn