import numpy as np
import torch



import const
import my_datasets
import my_models


model_dirs = sorted([path for path in const.RECOGNIZER_DIR.glob('*') if path != const.NPZ_DIR])
def get_latest_model_path(model_dir):
    return sorted(model_dir.glob('*.pt'), key=lambda p: int(p.stem))[-1]
print(model_dirs)


def infer_all(name, model, training):
    dataloader = my_datasets.get_dataloader(name, 64, training)

    model.eval()
    ys = list()
    labels = list()
    middles = list()
    with torch.no_grad():
        for img, _, label in dataloader:
            img = img.cuda()
            middle, y = model.forward(img, middle_required=True)
            middles.append(middle.cpu().numpy().astype(np.float16))
            pred = y.reshape((-1, len(const.ALL_CHAR_SET), const.MAX_CAPTCHA)).cpu()
            ys.append(torch.nn.functional.softmax(pred, dim=1).numpy().astype(np.float16))
            labels += label
    return np.concatenate(ys, axis=0), np.array(labels, dtype=str), np.concatenate(middles, axis=0)

def dump_npz(model_dir):
    model_path = get_latest_model_path(model_dir)
    model.load_state_dict(torch.load(str(model_path), map_location='cuda:0'))

    for data_name, (path_train, path_test) in my_datasets.DATA_DIRS.items():
        model_name = model_path.parent.name
        ys, labels, middles = infer_all((data_name, ), model, True)
        save_path = const.NPZ_DIR / '{}__{}-train.npz'.format(model_name, data_name)
        np.savez(save_path, ys=ys, labels=labels, middles=middles)

        ys, labels, middles = infer_all((data_name, ), model, False)
        save_path = const.NPZ_DIR / '{}__{}-test.npz'.format(model_name, data_name)
        np.savez(save_path, ys=ys, labels=labels, middles=middles)

if __name__ == '__main__':
    out_features = len(const.ALL_CHAR_SET) * const.MAX_CAPTCHA
    model = my_models.MyResNet(out_features)
    model.cuda()
    for model_dir in model_dirs:
        dump_npz(model_dir)
