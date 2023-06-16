import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mlp
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings(action='ignore')

def calculate_weighted_clusters(data, length, weights):
    sse = []
    silhouette_scores = []
    rmsle_scores = []

    for i in range(2, length):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        rmsle_scores.append(mean_squared_error(data, kmeans.cluster_centers_[kmeans.labels_]))

    sse_diff = np.diff(sse)
    optimal_cluster_elbow = np.argmin(sse_diff) + 2
    optimal_cluster_silhouette = np.argmax(silhouette_scores) + 2
    optimal_cluster_rmsle = np.argmin(rmsle_scores) + 2

    weighted_clusters = [
        optimal_cluster_elbow * weights[0],
        optimal_cluster_silhouette * weights[1],
        optimal_cluster_rmsle * weights[2]
    ]
    
    final_optimal_cluster = round(sum(weighted_clusters) / sum(weights))
    print("final_optimal_cluster : ",final_optimal_cluster)
    return final_optimal_cluster

def dr_outlier(df):
    quartile_1 = df.quantile(0.25)
    quartile_3 = df.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (df < (quartile_1 - 1.5 * IQR)) | (df > (quartile_3 + 1.5 * IQR))
    
    condition = condition.any(axis=1)
    search_df = df[condition]  
    return search_df
    
    
def cluster_kmeans(df, target):
    # target column을 제외한 dataframe 생성
    df_without_target = df.drop(labels=target, axis=1)
    
    # NaN 값이 있는지 확인하고, 있다면 적절한 값으로 채우기
    df_without_target = df_without_target.fillna(df_without_target.mean())

    # 데이터에 음수 값이 있는지 확인하고, 있다면 이를 처리
    if (df_without_target < 0).any().any():
        df_without_target = df_without_target - df_without_target.min().min()  # 모든 값을 양수로 만듭니다.

    #optimal_weights = grid_search_optimal_weights(df_without_target)
    n_clusters = calculate_weighted_clusters(df_without_target, 10,weights=[0.0, 0.2, 0.1])
    
    sc = StandardScaler()
    cc_scaled = sc.fit_transform(df_without_target)

    # 'dr_outlier' 함수가 DataFrame을 반환하도록 확인
    outliers = dr_outlier(df_without_target)
    if not isinstance(outliers, pd.DataFrame):
        raise ValueError("dr_outlier function should return a DataFrame")

    df_without_target['outlier'] = df_without_target.index.isin(outliers.index)

    kmeans = KMeans(n_clusters, random_state=0)
    clusters = kmeans.fit(cc_scaled)

    df_without_target['cluster'] = clusters.labels_

    # 이상치를 다시 클러스터링합니다.
    outliers_scaled = sc.transform(outliers)
    outlier_kmeans = KMeans(n_clusters, random_state=0)
    outlier_clusters = outlier_kmeans.fit(outliers_scaled)

    # 이상치 클러스터 번호를 'o_번호' 형식으로 변경하고, 원래 데이터 프레임에 추가합니다.
    outlier_cluster_labels = [100 + label for label in outlier_clusters.labels_]
    df_without_target.loc[outliers.index, 'cluster'] = outlier_cluster_labels
    df_without_target = df_without_target.drop(columns='outlier',axis=1)

    df_without_target.groupby('cluster').count()
    
    # 원래의 df에 cluster 정보를 추가합니다.
    df['cluster'] = df_without_target['cluster']
    
    return df

def record_partioning_sampling(df_clu,label_col):
  X = df_clu.drop(columns=[label_col])
  y = df_clu[label_col]
  
  # 데이터 분할
  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25, random_state=10)
  
  # SMOTE의 k-neighbors 값을 조절
  smote_neighbors = min(5, len(X_train) - 1)

  # 클래스별 샘플 수 확인
  class_counts = y_train.value_counts()

  # 적어도 6개 이상의 샘플이 있는 클래스만 추출
  sufficient_samples_classes = class_counts[class_counts >= 6].index

  # 충분한 샘플이 있는 클래스에 대해서만 SMOTE 적용
  if len(sufficient_samples_classes) > 0:
      smote = SMOTE(random_state=0, k_neighbors=smote_neighbors)
      X_train_clu, y_train_clu = smote.fit_resample(X_train[y_train.isin(sufficient_samples_classes)], y_train[y_train.isin(sufficient_samples_classes)])
  else:
      X_train_clu, y_train_clu = X_train, y_train  # SMOTE 적용하지 않음

  print('2차 SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_clu.shape, y_train_clu.shape)
  print('2차 SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_clu).value_counts())
  
  df_smote_cluster = pd.concat([X_train_clu, y_train_clu], axis=1)
  
  return df_smote_cluster

def record_clustering(raw_data,label,save_name):
  df = pd.read_csv(raw_data)
  df_record_clustering = cluster_kmeans(df,label)
  df_record_clustering = record_partioning_sampling(df_record_clustering,label)
  df_record_clustering.to_csv(save_name,index=False)