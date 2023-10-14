from multiprocessing import Pool

import numpy as np
from sklearn.manifold import TSNE



def task(num):

    data = np.random.rand(32, 32)

    while True:
        TSNE( n_components=2,
              learning_rate='auto',
              init='random',
              n_jobs=-1,
              ).fit_transform(data)



if __name__ == '__main__':

    # 8进程就可以干满 5950x
    # 内核时间80%以上，主要测试cpu科学计算稳定性
    p = Pool(processes=16)
    p.map(task, list(range(16)))
    p.close()
    p.join()
