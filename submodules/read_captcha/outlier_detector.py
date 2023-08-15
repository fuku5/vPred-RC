import numpy as np
import re

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class InstanceConfidenceCalculator():
    def __init__(self, train_known_positive, test_y):
        train_known_positive = train_known_positive.max(axis=1).prod(axis=1, keepdims=True)
        self.test_y = test_y.max(axis=1).prod(axis=1, keepdims=True)
        #train_known_negative = train_known_negative.max(axis=1).prod(axis=1, keepdims=True)

        self._positive = train_known_positive
        #self._negative
        #self.range_max = train_known_positive.mean() + 2.0 * train_known_positive.std()
        #self.range_min = train_known_negative.mean() - 2.0 * train_known_negative.std()
        #assert self.range_max > self.range_min
        # x: min, max -> y: 0, 1
        # y = ax + b
        # 1 = max*a + b
        # 0 = min*a + b
        # 1 = (max - min) * a
        # a = 1 / (max - min)
        # b = -min*a
        #   = - min / (max - min)
        self.range_max = self.test_y.max()
        self.range_min = self.test_y.min()
        

    
    def transform_linear(self, x):
        a = 1 / (self.test_y.max() - self.test_y.min())
        b = -self.test_y.min() * a
        return a * x + b
    
    def calc_score(self, x):
        #prob_pos = np.exp(self._gm_pos.predict_proba(x.max(axis=1).prod(axis=1, keepdims=True)))
        #prob_neg = np.exp(self._gm_neg.predict_proba(x.max(axis=1).prod(axis=1, keepdims=True)))
        #return prob_pos / (prob_pos + prob_neg)
        l = x.max(axis=1).prod(axis=1)
        if False:
            mean, sigma = self._positive.mean(), self._positive.std()
            l = x.max(axis=1).prod(axis=1)
            scores =  (l - mean) / (10 * sigma) * 0.25 + 0.75
            scores[scores > 1] = 1
            scores[scores < 0] = 0
            return scores
        elif False:
            return self.test_y
        else:
            return self.transform_linear(l)
        #return x.max(axis=1).prod(axis=1)
        #return self._gm.predict_proba(x.max(axis=1).prod(axis=1, keepdims=True))
        #original_likelihoods = self._gm.score_samples(self._positive)
        #mean, sigma = original_likelihoods.mean(), original_likelihoods.std()
        #l = self._gm.score_samples(x.max(axis=1).prod(axis=1, keepdims=True))
        #return (l - mean) / (1. * sigma) * 0.25 + 0.75
        



class OutlierDetector():
    def __init__(self, known, unknown, n_dim_pca=6, max_n_cluster=6):
        #known = np.concatenate(known)
        #if len(unknown) != 0:
        #    unknown = np.concatenate(unknown)
        #else:
        #    unknown = np.empty((0, known.shape[1]))

        self._known = known
        self._unknown = unknown

        self.pca = PCA(n_dim_pca)
        all = np.concatenate([known, unknown])
        self.pca.fit(all)

        y_known = self.pca.transform(known)

        def calc_gm(n_cluster, y_known):
            gm = GaussianMixture(n_cluster)
            gm.fit(y_known)
            return gm

        self.gm = max([calc_gm(i, y_known) for i in range(1, max_n_cluster)], key=lambda gm: gm.bic(y_known))
        

    def _calc_known_data_distribution(self):
        likelihoods = self.calc_likelihood(self._known)
        #print(likelihoods.shape)
        return likelihoods.mean(), likelihoods.std()

    
    def calc_likelihood(self, x):
        x = self.pca.transform(x)
        return self.gm.score_samples(x)

    def calc_score(self, x):
        mean, sigma = self._calc_known_data_distribution()
        l = self.calc_likelihood(x)
        scores = (l - mean) / (2 * sigma) * 0.25 + 0.75
        scores[scores > 1] = 1
        scores[scores < 0] = 0
        return scores
        #return np.exp(self.gm.score_samples(self.pca.transform(x)))


    def plot(self, n_bin=100):
        from matplotlib import pyplot as plt
        plt.clf()
        fig = plt.figure(figsize=(4,3*3))
        ax = fig.add_subplot(3,1,1)

        print(self._calc_known_data_distribution())
        y_known = self.pca.transform(self._known)
        likelihoods_known = self.gm.score_samples(y_known)
        if self._unknown.shape[0] != 0:
            y_unknown = self.pca.transform(self._unknown)
            likelihoods_unknown = self.gm.score_samples(y_unknown)
            min_l = min(likelihoods_known.min(), likelihoods_unknown.min())
            max_l = max(likelihoods_known.max(), likelihoods_unknown.max())

            ax.hist(likelihoods_known, bins=n_bin, range=(min_l, max_l), alpha=0.5)
            ax.hist(likelihoods_unknown, bins=n_bin, range=(min_l, max_l), alpha=0.5)
            
            ax = fig.add_subplot(3, 1, 3)
            y_ = y_known.transpose(1,0)
            ax.scatter(y_[0], y_[1], marker=',', color='tab:blue')
            
            y_ = y_unknown.transpose(1,0)
            ax.scatter(y_[0], y_[1], marker=',', color='tab:orange')
        else:
            ax.hist(likelihoods_known, bins=n_bin, alpha=0.5)

            ax = fig.add_subplot(3, 1, 3)
            y_ = y_known.transpose(1,0)
            ax.scatter(y_[0], y_[1], marker=',', color='tab:blue')

        plt.show()

def test():
    import const
    model_dirs = sorted([path for path in const.RECOGNIZER_DIR.glob('*') if path != const.NPZ_DIR])

    outlier_detectors = dict()

    for d in model_dirs:
        model_name = d.name
        npz_paths = list(const.NPZ_DIR.glob('{}__*-train.npz'.format(model_name)))

        known = list()
        unknown = list()

        for npz_path in npz_paths:
            dataset_name = re.search(r'__(.*)-train', str(npz_path)).groups(0)[0]
            npz = np.load(npz_path)
            _, _, middles = npz['ys'], npz['labels'], npz['middles']
            if dataset_name in model_name:
                known.append(middles)
            else:
                unknown.append(middles)
        print(len(known), len(unknown))

        outlier_detectors[model_name] = OutlierDetector(known, unknown)

    for model_name, outlier_detector in outlier_detectors.items():
        print(model_name)
        outlier_detector.plot()

if __name__ == '__main__':
    test()