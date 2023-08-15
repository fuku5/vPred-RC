import random
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression

from submodules.read_captcha import ALL_CHAR_SET, NPZ_DIR, load_npz_all
from submodules.read_captcha import OutlierDetector, InstanceConfidenceCalculator

TARGET_MODELS = ['captcha-09az+capital-color']


ACS = np.array(ALL_CHAR_SET)

def init():
    global npz_all, outlier_detectors, instance_confidence_calculators, acc_estimators
    npz_all, outlier_detectors, instance_confidence_calculators = load_npz()
    acc_estimators = prepare_acc_estimators()

def load_npz():
    npz_all = dict()

    outlier_detectors = dict()
    instance_confidence_calculators = dict()
    for model_name in TARGET_MODELS:
        npz_train = load_npz_all(NPZ_DIR, model_name, True)
        npz_test = load_npz_all(NPZ_DIR, model_name, False)
        npz_all[model_name] = npz_test
        mid_known = list()
        mid_unknown = list()
        y_positive = list()
        y_negative = list()
        for npz in npz_train.values():
            if npz['known']:
                mid_known.append(npz['middles'])
                c = [''.join(ACS[y.argmax(axis=0)]) for y in npz['ys']]
                # use label information only from training dataset
                y_positive.append(npz['ys'][c == npz['labels']])
                y_negative.append(npz['ys'][c != npz['labels']])
            else:
                mid_unknown.append(npz['middles'])
        y_test_all = [value['ys'] for value in npz_test.values()]

        outlier_detectors[model_name] = OutlierDetector(
            np.concatenate(mid_known),
            np.concatenate(mid_unknown)
            )
        instance_confidence_calculators[model_name] = InstanceConfidenceCalculator(
            np.concatenate(y_positive),
            np.concatenate(y_test_all)
            )
    return npz_all, outlier_detectors, instance_confidence_calculators

def prepare_acc_estimators():
    acc_estimators = dict()
    for model_name in TARGET_MODELS:
        npz = load_npz_all(NPZ_DIR, model_name, True)
        ys_all = np.concatenate([n['ys'] for n in npz.values()])
        middles_all = np.concatenate([n['middles'] for n in npz.values()])
        labels = np.concatenate([n['labels'] for n in npz.values()])

        domain_confs = outlier_detectors[model_name].calc_score(middles_all)
        instance_confs = instance_confidence_calculators[model_name].calc_score(ys_all)
        #print(domain_confs.shape, instance_confs.shape)
        #confs = np.stack([domain_confs, instance_confs], axis=1)
        confs = instance_confs.reshape((-1,1))

        top1s = np.apply_along_axis(lambda x: ''.join(x), 1, ACS[ys_all.argmax(axis=1)])

        acc_estimators[model_name] = LogisticRegression(class_weight='balanced')
        acc_estimators[model_name].fit(confs, top1s == labels)
    return acc_estimators

def list_labels(dataset_name, model_name):
    return npz_all[model_name][dataset_name].keys()

def get_y(dataset_name, model_name, label, target=['y']):
    assert all([t in ['y', 'middle'] for t in target])
    index = np.where(npz_all[model_name][dataset_name]['labels'] == label)[0]
    assert len(index) != 0
    rtn = list()
    if 'y' in target:
        rtn.append(
            npz_all[model_name][dataset_name]['ys'][index[0]]
        )
    if 'middle' in target:
        rtn.append(
            npz_all[model_name][dataset_name]['middles'][index[0]]
        )
    return rtn

def calc_top1(y):
    return ''.join(ACS[y.argmax(axis=0)])

def infer_and_calc_confidence(dataset_name, model_name, label):
    y, middle = get_y(dataset_name, model_name, label, ['y', 'middle'])

    instance_confidence =  instance_confidence_calculators[model_name].calc_score(y[np.newaxis])
    domain_confidence = outlier_detectors[model_name].calc_score(middle[np.newaxis])
    return calc_top1(y), instance_confidence, domain_confidence



def _load_npz(dataset_name, model_name):
    npz_path = Path('npz') / '{}-train_dataloader_{}.npz'.format(dataset_name, model_name)
    npz = np.load(npz_path)
    ys, labels = npz['ys'], npz['labels']
    d = {label: y.astype(np.float32) for label, y in zip(labels, ys)}
    return d


if __name__ == '__main__':
    a = load_npz()