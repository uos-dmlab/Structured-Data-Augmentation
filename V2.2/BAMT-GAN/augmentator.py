import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mlp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# %matplotlib inline

from imblearn.over_sampling import SMOTE
import numpy as np

def sampling_func(data, n_sample):
    N = len(data)
    sample = data.sample(n=min(N, n_sample), random_state=42)
    return sample

from imblearn.over_sampling import SMOTE
import numpy as np


def augmentator(df_gan):
    X = df_gan.iloc[:, :-1]
    y = df_gan.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    smote = SMOTE(random_state=0)
    X_train_label, y_train_label = smote.fit_resample(X_train, y_train)

    bamtgan = pd.concat([X_train_label, y_train_label], axis=1)
    return bamtgan


def new_balanced_sampling_and_augmentation(df, df_gan, augmentation_ratio, target):
    df_gan = df_gan.drop('cluster', axis=1)
    df_gan = df_gan.drop_duplicates()

    # 총 샘플 수 계산
    total_samples = int(len(df) * (1 + augmentation_ratio))

    # 최종 데이터셋에서 각 클래스별로 가져야 할 샘플 수 계산
    final_samples_per_class = total_samples // 2  # 50:50의 클래스 비율을 위해

    # 계산된 샘플 수에 따라 원본 및 증강 데이터에서 각 클래스별로 샘플 추출
    sampled_data = []
    for class_label in df[target].unique():
        original_class_data = df[df[target] == class_label]
        augmented_class_data = df_gan[df_gan[target] == class_label]

        # 원본 데이터는 그대로 가져가고, 증강 데이터에서 필요한 만큼만 샘플링
        n_samples_needed_from_augmented = max(0, final_samples_per_class - len(original_class_data))
        sampled_augmented_data = augmented_class_data.sample(n=n_samples_needed_from_augmented, replace=True, random_state=42)
        
        # 원본 데이터와 샘플링한 증강 데이터를 병합
        sampled_class_data = pd.concat([original_class_data, sampled_augmented_data])
        sampled_data.append(sampled_class_data)

    # 모든 클래스에 대해 병합된 데이터를 다시 병합
    bamtgan = pd.concat(sampled_data)

    return bamtgan

def new_ml_eval2(df, target):
    X = df.drop(labels=target, axis=1)
    y = df[target]

    # Add additional models
    models = [
        ('SVM', SVC(kernel='rbf', C=10, probability=True)),
        ('DT', tree.DecisionTreeClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('LR', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ('RF', RandomForestClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(eval_metric='mlogloss')),
        ('LightGBM', LGBMClassifier()),
        ('CatBoost', CatBoostClassifier(verbose=0)),
        ('NaiveBayes', GaussianNB()),
        ('LinearSVM', make_pipeline(StandardScaler(), LinearSVC())),
        ('AdaBoost', AdaBoostClassifier()),
        ('MLP', MLPClassifier(max_iter=1000))
    ]


    scoring = {'accuracy': make_scorer(accuracy_score),
               'auc': make_scorer(roc_auc_score),
               'f1': make_scorer(f1_score)}

    result_dict = {}
    for name, model in models:
        #print(name)
        kfold = KFold(n_splits=5, shuffle=True, random_state=10)
        cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        '''
        print('Accuracy : mean {0:.2f}, std {1:.2f} | AUC : mean {2:.2f}, std {3:.2f} | F1-Score : mean {4:.2f}, std {5:.2f}'.format(
            cv_results['test_accuracy'].mean()*100, cv_results['test_accuracy'].std()*100,
            cv_results['test_auc'].mean()*100, cv_results['test_auc'].std()*100,
            cv_results['test_f1'].mean()*100, cv_results['test_f1'].std()*100))
        result_dict[name] = [cv_results['test_accuracy'].mean()*100, cv_results['test_accuracy'].std()*100,
                              cv_results['test_auc'].mean()*100, cv_results['test_auc'].std()*100,
                              cv_results['test_f1'].mean()*100, cv_results['test_f1'].std()*100]
        '''
    return result_dict

from sklearn.impute import SimpleImputer

def ratio_eval(df, df_fake, target,name):

    imputer = SimpleImputer(strategy='mean')
    result_df = pd.DataFrame()
    orig_result_dict = new_ml_eval2(df, target)

    # Save the performance of the original data
    for key, value in orig_result_dict.items():
        result_df.loc['Original', key+'_accuracy_mean'] = value[0]
        result_df.loc['Original', key+'_accuracy_std'] = value[1]
        result_df.loc['Original', key+'_auc_mean'] = value[2]
        result_df.loc['Original', key+'_auc_std'] = value[3]
        result_df.loc['Original', key+'_f1_mean'] = value[4]
        result_df.loc['Original', key+'_f1_std'] = value[5]

    # Save the performance of the augmented data
    #augmentation_ratio = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
    augmentation_ratio = [0.2]
    for i in augmentation_ratio:
        #print(str(i*100)+'%------------------')
        aug_ratio_df = new_balanced_sampling_and_augmentation(df, df_fake, i, target)
        aug_ratio_df.to_csv('./augmented_data/'+str(name)+'/'+str(i)+name+'_bamtgan.csv',index=False)
        aug_ratio_df[aug_ratio_df.columns] = imputer.fit_transform(aug_ratio_df)
        result_dict = new_ml_eval2(aug_ratio_df,target)
        
        for key, value in result_dict.items():
            result_df.loc[str(i*100)+'%', key+'_accuracy_mean'] = value[0]
            result_df.loc[str(i*100)+'%', key+'_accuracy_std'] = value[1]
            result_df.loc[str(i*100)+'%', key+'_auc_mean'] = value[2]
            result_df.loc[str(i*100)+'%', key+'_auc_std'] = value[3]
            result_df.loc[str(i*100)+'%', key+'_f1_mean'] = value[4]
            result_df.loc[str(i*100)+'%', key+'_f1_std'] = value[5]
            
        aug_ratio_df[target].value_counts()

    # Add augmentation ratio as a separate column at the beginning
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'Augmentation Ratio'}, inplace=True)
    result_df.set_index('Augmentation Ratio', inplace=True)
    
    return result_df

from sklearn.preprocessing import LabelEncoder

def label_encoding_df(df):
  df = df.apply(LabelEncoder().fit_transform)
  return df

def augmentator(data,df_raw,df_record_gan,label,name):
  df_raw = label_encoding_df(pd.read_csv(df_raw))
  df_fake = pd.read_csv(df_record_gan)
  df_result = ratio_eval(df_raw,df_fake,label,data)
  df_result.to_csv('./result_metrics/'+str(name)+'_result.csv')