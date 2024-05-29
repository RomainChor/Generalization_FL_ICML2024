import idx2numpy
import numpy as np
from sklearn.datasets import (
    make_regression, make_friedman1, fetch_california_housing
)
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split




def load_binary_mnist(params, class1=1, class2=6, path=""):
    X = idx2numpy.convert_from_file(path+'train-images.idx3-ubyte')
    y = idx2numpy.convert_from_file(path+'train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file(path+'t10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file(path+'t10k-labels.idx1-ubyte')

    y = np.squeeze(y)
    y_test = np.squeeze(y_test)
    
    idxs = ((y == class1) + (y == class2)).nonzero()
    idxs_test = ((y_test == class1) + (y_test == class2)).nonzero()
    X = X[idxs]
    y = y[idxs]
    X_test = X_test[idxs_test]
    y_test = y_test[idxs_test]
    
#     y[y == class1] = -1
#     y[y == class2] = 1
#     y_test[y_test == class1] = -1
#     y_test[y_test == class2] = 1

    # Normalization
    X = (X/255. - 0.1307)/0.3081 
    X_test = (X_test/255. - 0.1307)/0.3081
    
    X = np.array([elm.ravel() for elm in X])
    X_test = np.array([elm.ravel() for elm in X_test])

    if params["proj_dim"] > X.shape[1]:
        kernel = RBFSampler(gamma=params["gamma"], n_components=params["proj_dim"])
        kernel.fit(np.concatenate((X, X_test), axis=0))
        X_proj = kernel.transform(X)
        X_test_proj = kernel.transform(X_test)
        
        return {"X":X_proj, "y":y, "X_test":X_test_proj, "y_test":y_test}
    
    return {"X":X, "y":y, "X_test":X_test, "y_test":y_test}



def make_classification_data(params, loaded_data=None):
    if params["name"] == "mnist":
        data = loaded_data.copy()
        if params["N"] <= data["X"].shape[0]: # Sampling
            data["X"], _, data["y"], _ = train_test_split(
                data["X"], 
                data["y"], 
                train_size=params["N"], 
                random_state=params["seed"],
                stratify=data["y"]
            )
        else:
            print("N is greater than the dataset size. The whole dataset is loaded.")
        
        
        return data



def make_regression_data(params):
    N_samples = params["N"]+params["N_test"]
    if params["name"] == "default":
        X, y = make_regression(
            n_samples=N_samples, 
            n_features=params["d"],
            n_informative=params["d"]//2,
            bias=params["bias"],
            noise=params["noise"],
            shuffle=True,
            random_state=params["seed"]
        )
    elif params["name"] == "friedman":
        X, y = make_friedman1(
            n_samples=N_samples,
            n_features=params["d"],
            noise=params["noise"],
            random_state=params["seed"]
        )
    elif params["name"] == "gaussian":
        X = np.random.normal(params["mean"], params["std"], (N_samples, params["d"]))
        W = np.random.normal(params["mean"], params["std"], params["d"])
        y = np.dot(X, W) 
        if params["noise"] > 0:
            y += params["noise"]*np.random.normal(0, 1.0, N_samples)
    elif params["name"] == "california":
        dataset = fetch_california_housing()
        if N_samples <= len(dataset.target):
            X, _, y, _ = train_test_split(
                dataset.data, 
                dataset.target, 
                train_size=N_samples,   
                random_state=params["seed"]
            )
        else:
            X, y = dataset.data, dataset.target
            print("N+N_test is greater than the dataset size. The whole dataset is loaded.")
    else:
        raise ValueError("params['name'] is not recognized!")
    
    X, X_test, y, y_test = train_test_split(
        X, 
        y, 
        test_size=params["N_test"], 
        random_state=params["seed"]
    )
    X_test += np.random.normal(0, 0.2, X_test.shape)
    
    if params["scale"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
    
    return {"X":X, "y":y, "X_test":X_test, "y_test":y_test}
