import time
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class IcartoonFaceScore:
    def __init__(self, icartoonface_rectest_info_txt, feat_size=22500):
        super().__init__()
        self.icartoonface_rectest_info_txt = icartoonface_rectest_info_txt
        self.feat_size = feat_size

    def compute_score(self, bin_path):
        stars = np.fromfile(bin_path, dtype=float)
        feats, d = [], len(stars) // self.feat_size
        for x in range(self.feat_size):
            feat = stars[x * d:(x + 1) * d]
            feats.append(feat)

        distance = pairwise_distances(feats, feats, metric='cosine', n_jobs=-1)
        dis_mat = np.argsort(distance, axis=0).T
        imgpaths, imgpath_classids = [], []
        correct_num, total_num = 0, 0
        with open(self.icartoonface_rectest_info_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line_info = line.strip().split()
                if len(line_info) == 6:
                    imgpaths.append(line_info[0])
                    imgpath_classids.append(line_info[-1])
                if len(line_info) == 2:
                    imgpath1, imgpath2 = line_info[0], line_info[1]
                    idx1, idx2 = imgpaths.index(imgpath1), imgpaths.index(imgpath2)
                    idx1_classid = imgpath_classids[idx1]
                    for idx_var in dis_mat[idx1]:
                        idx_classid = imgpath_classids[idx_var]
                        if idx_classid == idx1_classid and idx_var == idx2:
                            correct_num += 1
                        elif not idx_classid == '-1':
                            continue
                        else:
                            break
                    total_num += 1
                    # print('{}/{}, accuracy: {}'.format(correct_num, total_num, 100.0*correct_num/total_num))
        return 100.0*correct_num/total_num

if __name__ == '__main__':
    # initialization
    icartoonFaceScore = IcartoonFaceScore('icartoonface_rectest_info.txt')

    # compute score
    s_time = time.time()
    input_bin_path = 'input.bin'
    print(icartoonFaceScore.compute_score(input_bin_path))
    print('total time cost: {}s'.format(time.time()-s_time))
