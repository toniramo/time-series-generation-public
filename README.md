# Analysis of multivariate time-generation methods.
Public repository to provide auxiliary material for the "Analysis of multivariate time-series generation methods" project. The project is part of the course "Analyzing the changing world" (LSI32004) of the University of Helsinki. 

Abstract:
> This paper analyzes three GAN-based methods for generating multivariate time series: TimeGAN,
RCGAN, and RCWGAN. Using five diverse real-life datasets, we evaluate the performance of each
model through numerical metrics and visualizations. Our results indicate that no single method
consistently outperforms the others across all datasets. Instead, the effectiveness of each model
is dependent on the specific characteristics of the dataset and the application requirements. This
study highlights the importance of tailored model selection and thorough hyperparameter tuning
in achieving high-quality synthetic time series generation. Future work should explore additional
methods, metrics, and more robust tuning strategies.

The implementation is based on the repository [Conditional-Sig-Wasserstein-GANs](https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs). Additionally, the visualizations of t-SNE and PCA are based on the script available at [TimeGAN](https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/visualization_metrics.py).

Contents:
- [Project report](Time-series_generation_Rämö.pdf)
- [Implementation](./src)
- [Numerical results](./src/numerical_results/) of the best runs
- [Datasets](./src/datasets) and their preprocessing notebooks
