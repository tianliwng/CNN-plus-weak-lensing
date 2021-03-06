# P212:Cosmology--Final Project
In this project, we partially follow the work done in Ref.[1](#f1) [2](#f2) to analyze the constraints on ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B80%7D%20%5Cbg_white%20%5CLARGE%20%5COmega_m) and ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B80%7D%20%5Cbg_white%20%5CLARGE%20%5Csigma_8) from weak lensing maps by using power spectra, peak counts, and CNN predictions. We show that CNNs have the potential to extract more information than summary statistics.

The files in this repo work as follows: 

1. in fullCNNtrain_gpu.py we implement, train (with validation), and save the CNN
2. fullNet_testing.ipynb generates the network predictions
3. likelihood_all.ipynb computes the power spectrum and peak counts, as well as calculates and visualizes the constraints on cosmological models derived from the different methods (CNN, power spectrum, and peak counts)
4. check_maps_and_preds.ipynb checks how downsizing influences the appearance of a map, and checks if the CNN predictions for each map have a Gaussian distribution.



<a name="f1">1</a>: Ribli et al., arXiv: 1902.03663

<a name="f2">2</a>: Gupta et al., arXiv: 1802.01212
