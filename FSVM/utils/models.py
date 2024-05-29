import random
import time
import numpy as np
import pandas as pd
from tqdm import tnrange
from sklearn.linear_model import SGDRegressor, SGDClassifier
import sklearn.metrics as metrics

from utils.generalization.dataloaders import make_classification_data




def theta_risk(outputs, targets, theta=0):
    """
    Computes the 0-1 margin risk.
    Args:
        outputs (array-like): outputs of the decision function of the classifier.
        targets (array-like): targets to predict.
        classes (tuple-like): labels of the classes (binary classification).
        theta (float >= 0): margin value, default = 0.
    Returns: risk value (float)
    """
    if theta < 0:
        raise ValueError("theta must be >= 0!")
        
    labels = targets.copy()
    classes = np.unique(labels)
    labels[labels == classes[0]] = 0
    labels[labels == classes[1]] = 1
    
    return (2*(labels-0.5)*outputs.transpose() < theta).mean()



def load_local_model(params):
    if params["task"] == "regression":
        mod = SGDRegressor
    else:
        mod = SGDClassifier
    model = mod(
        loss=params["loss"],
        penalty="l2",
        max_iter=1,
        fit_intercept=True,
        shuffle=True,
        random_state=params["seed"],
        learning_rate='adaptive',
        tol=1e-2,
        n_iter_no_change=10,
        eta0=params["lr"]
    )

    return model



def load_global_model(params):
    if params["task"] == "regression":
        mod = SGDRegressor
    else:
        mod = SGDClassifier
    model = mod(
        loss=params["loss"],
        penalty="l2",
        max_iter=1,
        fit_intercept=True,
        shuffle=False,
        random_state=params["seed"],
        learning_rate='constant',
        eta0=1e-32 
    )
    
    return model



def get_loss_pred_fn(name):
    if name == "squared_error":
        return metrics.mean_squared_error, "predict"
    elif name == "hinge":
        return metrics.hinge_loss, "decision_function"
    elif name == "log":
        return metrics.log_loss, "predict_proba"
    else:
        raise ValueError()



def generate_noise(d, generator=None):
    """Generate the uniform perturbation to SGD."""
    if generator is None:
        generator = np.random.default_rng()
    x = generator.standard_normal(d)
    
    return x / np.linalg.norm(x)


        
class Client:
    """
    Class emulating each client of the network. 
    """
    def __init__(self, X, y, params, generator=None):
        self.X = np.array_split(X, params["n_rounds"])
        self.y = np.array_split(y, params["n_rounds"])
        self.X_place, self.y_place = self.compute_placeholders(X, y)
        self.model = load_local_model(params)
        self.epochs = params["client_epochs"]
        self.round = 0
        self.generator = generator if generator is not None else np.random.default_rng(0)
        
    def compute_placeholders(self, X, y):
        classes = np.unique(y)
        idx_1 = np.where(y == classes[0])[0][0]
        idx_2 = np.where(y == classes[1])[0][0]

        return np.array([X[idx_1], X[idx_2]], ndmin=2), np.array([y[idx_1], y[idx_2]])
        
    def initial_fit(self, coef=None, intercept=None):
        self.model.fit(
            np.concatenate((self.X[self.round], self.X_place), axis=0), 
            np.concatenate((self.y[self.round], self.y_place)), 
            sample_weight=np.concatenate(
                (np.repeat(1., len(self.y[self.round])), np.repeat(0, 2)), 
                axis=0
            ),
            coef_init=coef, 
            intercept_init=intercept
        )

    def partial_fit(self):
        # idxs = np.arange(len(self.y[self.round]))
        idxs = self.generator.permutation(len(self.y[self.round])) # Shuffle at each local "epoch"
        self.model.partial_fit(self.X[self.round][idxs], self.y[self.round][idxs])
        
    def run_SGD_iterations(self):
        for e in range(self.epochs-1):
            self.partial_fit()



class Server:
    """
    Class emulating the central server. Instructs the clients to train their models, 
    then aggregates their parameters and computes risks values.
    """
    def __init__(self, X_test, y_test, params, X_test_iid=None):
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_iid = X_test_iid
        self.X_placeholder = X_test[:10]
        self.y_placeholder = y_test[:10]

        # self.global_coef_ = None
        self.global_coef_ = generate_noise(X_test.shape[1]) # Uniform on the unit ball
        self.global_intercept_ = None
        self.n_clients = 0
        self.frac_iid = params["frac_iid"]
        self.distribution = []   
        self.test_risks, self.emp_risks = [], []
        self.barnes_risks = []
        self.global_model = load_global_model(params) 
        self.loss_fn, pred = get_loss_pred_fn(params["loss"])
        self.pred_fn = getattr(self.global_model, pred)
        self.round = 0

    def add_participant(self, participant):
        self.distribution = np.append(self.distribution, participant)
        self.n_clients += 1

    def _aggregate_models(self):
        temp_coef = 0
        temp_inter = 0
        for client in self.distribution:
            temp_coef += client.model.coef_
            temp_inter += client.model.intercept_
        temp_coef /= self.n_clients
        temp_inter /= self.n_clients

        self.global_coef_ = temp_coef
        self.global_intercept_ = temp_inter
        
        self.global_model.fit( 
            X=self.X_placeholder,
            y=self.y_placeholder,
            coef_init=self.global_coef_,
            intercept_init=self.global_intercept_
        ) # Sets global model's parameters to the aggregated ones
    
    def compute_emp_risk(self):
        emp_risk = 0
        for client in self.distribution:
            l = 0
            for r in range(client.round+1):
                outputs = self.pred_fn(client.X[r])
                l += self.loss_fn(client.y[r], outputs)
            l /= client.round+1
            emp_risk += l
        emp_risk /= self.n_clients
        
        return emp_risk
    
    def compute_barnes_risk(self):
        barnes_risk = 0
        for client in self.distribution:
            outputs = self.pred_fn(client.X[client.round])
            barnes_risk += self.loss_fn(client.y[client.round], outputs) 
        barnes_risk /= self.n_clients
        
        return barnes_risk
    
    def compute_test_risk(self):
        outputs = self.pred_fn(self.X_test)
        
        return self.loss_fn(self.y_test, outputs)
    
    def compute_01_risks(self):
        if self.X_test_iid is not None:
            risk_1 = (1 - (self.global_model.score(self.X_test_iid, self.y_test)))*self.frac_iid
            risk_2 = (1 - (self.global_model.score(self.X_test, self.y_test)))*(1 - self.frac_iid)
            risk = risk_1 + risk_2
        else:
            risk = 1 - self.global_model.score(self.X_test, self.y_test)
        
        emp_risk = 0
        for client in self.distribution:
            l = 0
            for r in range(client.round+1):
                l += 1 - self.global_model.score(client.X[r], client.y[r])
            l /= client.round+1
            emp_risk += l
        emp_risk /= self.n_clients
        
        return emp_risk, risk
        
    def _update_risks(self):
        self.test_risks.append(self.compute_test_risk())
        # self.barnes_risks.append(self.compute_barnes_risk())

    def run_round(self):
        for client in self.distribution:
            client.round = self.round
            client.initial_fit(coef=self.global_coef_, intercept=self.global_intercept_)
            client.run_SGD_iterations()

        self._aggregate_models()
        self._update_risks()
        self.round += 1
        
    
        
def federated_learning(data, params, K=2, generator=None):
    """
    Runs the distributed learning setup. 
    Splits data into client datasets, creates clients and server instances and launch training. 
    Args:
        data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
        params (dict): parameters for training.
        K (int >= 2): number of clients, default = 2.
    """
    if K < 2:
        raise ValueError("K must be >= 2!")
    if params["n_rounds"] < 1:
        raise ValueError("params['n_rounds'] must be >= 1!")
        
    X_list = np.array_split(data["X"], K)
    y_list = np.array_split(data["y"], K)
    X_test_iid = None
    d = data["X"].shape[1]
    if params["frac_iid"] > 0: 
        noise = np.random.normal(0, params["iid_std"], d)
        for k in range(int(K*params["frac_iid"])):
            X_list[k] += noise
        X_test_iid = data["X_test"] + noise
    server = Server(data["X_test"], data["y_test"], params, X_test_iid)
    for k in range(K):
        server.add_participant(Client(X_list[k], y_list[k], params))

    stream = tnrange(params["n_rounds"])
    for r in stream:
        server.run_round()
        stream.set_description ("Round {0} achieved.".format(r+1))
        
    return server



# def centralized_learning(data, params):
#     """
#     Runs the centralized learning setup. 
#     Args:
#         data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
#         params (dict): parameters for training.
#     """
#     model = load_local_model(params)
#     model.fit(data["X"], data["y"])
    
#     loss_fn, pred = get_loss_pred_fn(params["loss"])
#     pred_fn = getattr(model, pred) 
#     outputs = pred_fn(data["X"])
#     emp_risk = loss_fn(data["y"], outputs)
#     if params["task"] == "classification":
#         emp_risk_01 = 1 - model.score(data["X"], data["y"])
#         print("Epoch 1 done. Emp risk: {0:.4f}".format(emp_risk_01))
#     else:
#         print("Epoch 1 done. Emp risk: {0:.4f}".format(emp_risk))
    
#     stream = tnrange(2, params["epochs"]+1)
#     for e in stream:
#         idxs = np.random.permutation(len(data["y"])) # Shuffle at each epoch 
#         model.partial_fit(data["X"][idxs], data["y"][idxs])
#         outputs = pred_fn(data["X"][idxs])
#         emp_risk = loss_fn(data["y"][idxs], outputs)
#         if params["task"] == "classification":
#             emp_risk_01 = 1 - model.score(data["X"][idxs], data["y"][idxs])
#             stream.set_description("Epoch {0} done. Emp risk: {1:.4f}".format(e, emp_risk_01))
#         else:
#             stream.set_description("Epoch {0} done. Emp risk: {1:.4f}".format(e, emp_risk))
# #         if emp_risk <= params["min_risk"]:
# #             break
        
#     outputs = pred_fn(data["X_test"])
#     risk = loss_fn(data["y_test"], outputs)
#     if params["task"] == "classification":
#         risk_01 = 1 - model.score(data["X_test"], data["y_test"])
#         return emp_risk, risk, emp_risk_01, risk_01

#     return emp_risk, risk



def compute_bound(n, K, R, q, theta=1.0, B=1.0):
    c1 = 252*(B**2)*np.log(n*K*np.sqrt(K))/(2*n*K**2*theta**2)
    f = lambda r, q: 2*q**(2*(R-r))*np.log(np.maximum( 2*K*theta/(7*B*q**(R-r)), 2 ))
    g = 2*np.log(np.maximum( 2*K*theta/(7*B), 2 ))
    c2 = np.sum([np.minimum(f(r, q), g) for r in range(1, R+1)])

    return np.sqrt(c1*c2)



def generalization_rounds(K, R_values, data_params, params, mnist, MC=5):
    df = pd.DataFrame(0, 
                    index=R_values,
                    columns=["fed_emp_risks_01", "fed_risks_01"])
    for m in range(MC):
        print("m =", m+1)
        seed = np.random.randint(0, MC)
        data_params["seed"] = seed
        params["seed"] = seed
        
        data = make_classification_data(data_params, mnist)
        
        fed_emp_risks_01, fed_risks_01 = [], []
        fed_times = []
        for R in R_values:
            params["n_rounds"] = R 
            
            start = time.time()
            server = federated_learning(data, params, K)
            end = time.time()
            fed_times.append(end-start)
            print("Federated learning setup total runtime: {0:.3f}s.".format(end-start))
            
            fed_emp_risk_01, fed_risk_01 = server.compute_01_risks()
            print("Empirical risk/Test risk: {0:.3f}/{1:.3f}".format(fed_emp_risk_01, fed_risk_01))
            fed_emp_risks_01.append(fed_emp_risk_01)
            fed_risks_01.append(fed_risk_01)
            print("=============================")
        df["fed_emp_risks_01"] += fed_emp_risks_01
        df["fed_risks_01"] += fed_risks_01
        print("===============================================")

    df /= MC
    df["fed_gen"] = df["fed_risks_01"] - df["fed_emp_risks_01"]

    return df



def compute_q(K, data_params, params, mnist, MC=10):
    generator = np.random.default_rng(0)
    data = make_classification_data(data_params, mnist)

    X_list = np.array_split(data["X"], K)
    y_list = np.array_split(data["y"], K)

    q_values = np.zeros(params["n_rounds"])
    for _ in range(MC):
        seed = np.random.randint(0, MC)
        data_params["seed"] = seed
        params["seed"] = seed

        server = Server(data["X_test"], data["y_test"], params)
        noisy_server = Server(data["X_test"], data["y_test"], params)
        for k in range(K):
            server.add_participant(Client(X_list[k], y_list[k], params, generator))
            noisy_server.add_participant(Client(X_list[k], y_list[k], params, generator))
        # server.run_round()
        # noisy_server.run_round()
        coef = server.global_coef_.copy()
        noisy_coef = noisy_server.global_coef_.copy()

        qs = []
        for r in range(params["n_rounds"]):
            server.run_round()
            noisy_server.run_round()
            new_coef = server.global_coef_.copy()
            new_noisy_coef = noisy_server.global_coef_.copy()
            qs.append(np.linalg.norm(new_coef - new_noisy_coef)/np.linalg.norm(coef - noisy_coef))
            # print(np.linalg.norm(new_coef - new_noisy_coef)/np.linalg.norm(coef - noisy_coef))
            coef = new_coef.copy()
            noisy_coef = new_noisy_coef.copy()
        q_values += np.array(qs)
    q_values /= MC

    return q_values