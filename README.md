# Structured-Data-Augmentation

##Automatic Augmentation Technique of an Autoencoder-based Numerical Training Data(D-VAE)

This study aims to solve the problem of class imbalance in numerical data by using a deep learning-based Variational AutoEncoder and to improve the performance of the learning model by augmenting the learning data. We propose 'D-VAE' to artificially increase the number of records for a given table data. The main features of the proposed technique go through discretization and feature selection in the preprocessing process to optimize the data. In the discretization process, K-means are applied and grouped, and then converted into one-hot vectors by one-hot encoding technique. Subsequently, for memory efficiency, sample data are generated with Variational AutoEncoder using only features that help predict with RFECV among feature selection techniques. To verify the performance of the proposed model, we demonstrate its validity by conducting experiments by data augmentation ratio.



## V1. BAMT-GAN:  A Balanced Data Augmentation Techniqe for Tabular Data

This paper presents BAMTGAN, a novel data augmentation technique that addresses the class imbalance problem and prevents mode collapse by utilizing a modified DCGAN model and a new similarity loss to generate diverse and realistic tabular data. BAMTGAN encodes each column to produce a feature map for each record, which is then converted back to its original tabular form an intermediate image format. Experimental results demonstrate that BAMTGAN provides a more substantial improvement in developing high-quality predictive models than existing augmentation methods.


## V2. BAMT-GAN:  A Balanced Data Augmentation Techniqe for Tabular Data

In this paper, we propose a novel method, BAMT-GAN (Balanced Augmentation for Mixed Tabular GAN), aimed at enhancing the quality of data through augmentation, thereby increasing the volume of reliable training data for model building. This technique caters specifically to augmenting mixed data, inclusive of both numeric and categorical columns. 
The BAMT-GAN integrates generative models, clustering, and oversampling techniques to balance data. The unique features of our approach involve 'Record Clustering' where data bearing similar features are clustered and diversified through oversampling. 'Data Generation' employs a CNN-based GAN model to train table data, generating novel data. In 'Classification', a myriad of classification algorithms is amalgamated to derive accurate predictive values. Lastly, in 'Augmentation', stratified sampling is leveraged to construct a balanced dataset. 
To validate the efficacy of the proposed technique, experiments were conducted across diverse datasets from various domains including healthcare, marketing, finance, and environmental science. The experimental outcomes indicated that the BAMT-GAN efficiently mitigates the data imbalance issue and enhances the performance of classification models, outperforming existing data augmentation techniques. These findings suggest that data augmentation can effectively extend training data, aiding in the construction of more dependable predictive models.
