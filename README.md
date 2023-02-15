# Structured-Data-Augmentation

## Ver1. Automatic Augmentation Technique of an Autoencoder-based Numerical Training Data(D-VAE)

This study aims to solve the problem of class imbalance in numerical data by using a deep learning-based Variational AutoEncoder and to improve the performance of the learning model by augmenting the learning data. We propose 'D-VAE' to artificially increase the number of records for a given table data. The main features of the proposed technique go through discretization and feature selection in the preprocessing process to optimize the data. In the discretization process, K-means are applied and grouped, and then converted into one-hot vectors by one-hot encoding technique. Subsequently, for memory efficiency, sample data are generated with Variational AutoEncoder using only features that help predict with RFECV among feature selection techniques. To verify the performance of the proposed model, we demonstrate its validity by conducting experiments by data augmentation ratio.



## Ver2. Balanced Augmentation of Mixed type Tabular data => 연구 진행중
