# Structured-Data-Augmentation

## V1. Automatic Augmentation Technique of an Autoencoder-based Numerical Training Data(D-VAE)

This study aims to solve the problem of class imbalance in numerical data by using a deep learning-based Variational AutoEncoder and to improve the performance of the learning model by augmenting the learning data. We propose 'D-VAE' to artificially increase the number of records for a given table data. The main features of the proposed technique go through discretization and feature selection in the preprocessing process to optimize the data. In the discretization process, K-means are applied and grouped, and then converted into one-hot vectors by one-hot encoding technique. Subsequently, for memory efficiency, sample data are generated with Variational AutoEncoder using only features that help predict with RFECV among feature selection techniques. To verify the performance of the proposed model, we demonstrate its validity by conducting experiments by data augmentation ratio.



## V2. BAMT-GAN:  A Balanced Data Augmentation Techniqe for Tabular Data
This paper presents BAMTGAN, a novel data augmentation technique that addresses the issue of class imbalance and prevents mode collapse by utilizing a modified DCGAN model and a new similarity loss to generate diverse and realistic tabular data. BAMTGAN encodes each column to produce a feature map for each record, which is then converted back to its original tabular form from an intermediate image format. Experimental results demonstrate that BAMTGAN provides a more substantial improvement in the development of high-quality predictive models than existing augmentation methods.
