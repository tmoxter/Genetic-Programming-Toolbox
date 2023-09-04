from evolution import Evolution
from framework import Framework
from models.classicGP import ClassicGP
from models.seiaGP import SeiaGP
from models.pseudoGSM import PseudoGSM
from models.denseAE import DenseAutoencoder
from models.RnnVAE import RecurrentVarAutoEncoder
from models.trainers import TrainerDenseAE, TrainerRNNAE, TrainerRnnVAE
from models.syntaxHC import SyntaxHC
import torch
import joblib
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_breast_cancer

# --- example for breast cancer classification using novel (local) GP as well
#     as classic GP as baseline in a tensor-based framework --- #

config = {
"seed":42,
"n_problems":1,
"n_sample": 2000,
"alphabet": ["+", "-", "*", "/", "cos", "sin", "exp", "log", "3.14"],
"max_depth":3,
"tournament_size":2,
"var_operator":"subtree-mutation",
"pop_size":2000,
"max_gens":65, "max_time":None, "max_eval":None,
"n_epochs":None,
"compression":None,
"batch_size":None,
"l1_coef":None,
"lr":None,
"step_size": 0.1,
"n_runs":5,
"split":0.8,
}

# --- data preparation --- #
breast_cancer = load_breast_cancer()
xTrain, yTrain = breast_cancer["data"], breast_cancer["target"]
cut = int(config["split"] * len(xTrain))
xTrain, xTest = torch.tensor(xTrain[:cut,:], dtype=torch.float),\
                torch.tensor(xTrain[cut:,:], dtype=torch.float)

# --- (z-score) standardized data --- #
xTrain = (xTrain - xTrain.mean(dim=0)) / xTrain.std(dim=0)
xTest = (xTest - xTrain.mean(dim=0)) / xTrain.std(dim=0)
yTrain, yTest = torch.tensor(yTrain[:cut], dtype=torch.float),\
         torch.tensor(yTrain[cut:], dtype=torch.float)

# --- intialize framework --- #
framework = Framework(xTrain, config["alphabet"], config["max_depth"],
                    max_cpu_count=4)

# --- initialize model to choose variation operator --- #
#model = SeiaGP(framework, operator=config["var_operator"]) # ---> local GP (NISSP)
#model = PseudoGSM(framework, operator=config["var_operator"]) # ---> local GP (Pseudo-GSM)
#model = RecurrentVarAutoEncoder(framework, operator=config["var_operator"]) # ---> local GP (RNN-VAE) should be pretrained (see thesis)
model = ClassicGP(framework, operator=config["var_operator"]) # ---> classic GP

# --- run evolution --- #
model_run = []
for i in range(config["n_runs"]):

    evo = Evolution(framework, model, xTrain, yTrain, 
                    tournament_size=config["tournament_size"], step_size=config["step_size"],
                    population_size=config["pop_size"], max_gens=config["max_gens"], verbose=True,
                    prevent_revists=True, use_scaling=False)
    evo.evolve()
    model_run.append(evo.results)
    print("Run {} of {} finished on {}.".format(i+1, config["n_runs"], "breast_cancer"))

joblib.dump(model_run, "data/{}-{}-{}.joblib".format("GPm_d3", "breast_cancer",
        datetime.now().strftime("%m-%d"))
    )