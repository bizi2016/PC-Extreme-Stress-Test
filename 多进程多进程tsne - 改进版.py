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

    # 内核时间95%以上，主要测试cpu科学计算稳定性

    n = 4
    
    p = Pool(processes=n)
    p.map(task, list(range(n)))
    p.close()
    p.join()
