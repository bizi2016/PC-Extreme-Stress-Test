import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

import time

import warnings

# 将警告过滤器设置为'ignore'，表示忽略所有警告
warnings.filterwarnings('ignore')

####################
# 数据处理
####################

# 读取CSV文件
df = pd.read_csv('processed_data.csv')

# 取数据集
data = df.drop(columns=['class'])
label = df['class']

# 分割数据集
x_train, x_test, y_train, y_test = \
         train_test_split( data, label, test_size=0.2, stratify=label )

####################
# XGBoost
####################

# 计算正负样本比例
scale_pos_weight = np.sum(y_train==0) / np.sum(y_train==1)
print('scale_pos_weight =', scale_pos_weight)
print()

# 定义XGBoost GPU模型参数
params = {
    'objective': 'binary:logistic',
    # 'mlogloss': Multiclass logloss
    # 'auc': Receiver Operating Characteristic Area under the Curve
    'eval_metric': 'logloss',
    
    'scale_pos_weight': scale_pos_weight,

    'tree_method': 'gpu_hist',  # 使用GPU进行树的构建，不精确快速算法
    'predictor': 'gpu_predictor',  # 使用GPU进行预测
    
    'n_jobs': 16,  # 一个CPU填不满显卡
    # 'n_gpus': -1,  # 使用所有可用的GPU
}

# 初始化XGBoost分类器
gpu_clf = xgb.XGBClassifier(**params)

####################
# GridSearchCV
####################

# 定义GridSearchCV的参数网格
param_grid = {
    # 'learning_rate': [0.3, 0.1, 0.01, 0.001],  # 较小的学习率 默认 0.3
    # 'n_estimators': [100, 300, 500, 700, 1000],  # 增加迭代次数 默认 100
    'n_estimators': [100, 200, 300],  # 增加迭代次数 默认 100
    
    # 'max_depth': [3, 4, 5, 6, 7, 8],  # 较小的树深度 默认 6
    'max_depth': [3, 4, 5, 6],  # 较小的树深度 默认 6
    
    'subsample': [0.6, 0.8, 1.0],  # 可以尝试不使用全部样本 默认 1.0
    'colsample_bytree': [0.6, 0.8, 1.0],  # 控制每棵树使用的特征比例 默认 1.0
}

# 初始化GridSearchCV
grid_search = GridSearchCV( estimator=gpu_clf, param_grid=param_grid,
                            scoring='roc_auc', cv=4,  # cv参数表示交叉验证的折数（folds）
                            verbose=1,  n_jobs=16,
                            )

####################
# GridSearchCV训练
####################

# 记录开始时间
start_time = time.time()

# 运行GridSearchCV进行训练和搜索
grid_search.fit(x_train, y_train)

# 记录结束时间
end_time = time.time()

# 计算程序运行时间
execution_time = end_time - start_time
print('execution_time =', execution_time, 's')
print()

####################
# 训练
####################

# 获取最佳参数
best_params = grid_search.best_params_
print('best_params =', best_params)
print()

# 获取最佳模型
best_model = grid_search.best_estimator_

# 训练模型
best_model.fit(x_train, y_train)

####################
# 保存结果
####################

# 获取详细的搜索情况
report = pd.DataFrame(grid_search.cv_results_)

# 将结果保存为CSV文件
report.to_csv('grid_search_results.csv', index=False)

####################
# 保存模型
####################

import joblib

# 保存最佳模型
joblib.dump(best_model, 'best_xgboost_model.pkl')

# 加载模型
# loaded_model = joblib.load('best_xgboost_model.pkl')

####################
# 预测
####################

# 获取预测概率
y_proba = best_model.predict_proba(x_test)[:, 1]
y_pred =    y_proba >= 0.5

####################
# 三分评估
####################

# 计算准确率、F1分数和ROC-AUC值
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print('acc =', acc)
print('f1 =', f1)
print('roc_auc =', roc_auc)

####################
# ROC曲线
####################

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6), dpi=200)

plt.plot( fpr, tpr, color='darkorange', lw=2,
          label='ROC curve (area = {:.2f})'.format(roc_auc),
          )

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')  # 将图例显示在右下角

# plt.show()

plt.savefig('roc.png', dpi=200)
plt.close()

####################
# 提取特征重要性csv
####################

fi = best_model.feature_importances_

# 创建包含特征名的DataFrame
feature_names = data.columns  # data是训练集
fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})

# 按照重要性从大到小排序
fi_df = fi_df.sort_values(by='importance', ascending=False)

# 将DataFrame保存为CSV文件
fi_df.to_csv('grid_search_feature_importance.csv', index_label='index', index=True)

"""
scale_pos_weight = 3.1793774735265803

Fitting 4 folds for each of 108 candidates, totalling 432 fits
execution_time = 212.142315864563 s

best_params = {'colsample_bytree': 1.0, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0}

acc = 0.8366260620329614
f1 = 0.7175221238938053
roc_auc = 0.9297958382790333
"""











