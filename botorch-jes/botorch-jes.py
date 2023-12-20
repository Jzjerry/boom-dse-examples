import random
import torch
import sklearn
import numpy as np
import botorch
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier


from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning)

from botorch.acquisition.multi_objective.utils import (
    sample_optimal_points,
    random_search_optimizer,
    compute_sample_box_decomposition
)
from botorch.optim import optimize_acqf
from utils import *
torch.manual_seed(0)
np.random.seed(0)

class BOTorchOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        self.model = []
        self.x = []
        self.y = []
        self.n_inits = 30
        self.n_suggest = 1
        self.init = True
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()
        self.bounds = torch.Tensor(
            [np.min(self.microarchitecture_embedding_set, axis=0),
            np.max(self.microarchitecture_embedding_set, axis=0)+0.0001])
        self.lb = np.min(self.microarchitecture_embedding_set, axis=0)
        self.ub = np.max(self.microarchitecture_embedding_set, axis=0)+0.0001
        self.dim = len(self.microarchitecture_embedding_set[0])
        self.std_bounds = torch.zeros(2, self.dim)
        self.std_bounds[1] = 1

    def construct_microarchitecture_embedding_set(self):
        microarchitecture_embedding_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_embedding_set.append(
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(i)
                )
            )
        return np.array(microarchitecture_embedding_set)

    def get_ac(self, model, num_pareto_samples=10, num_pareto_points=10):

        optimizer_kwargs = {
            "pop_size": 2000,
            "max_tries": 10,
        }

        ps, pf = sample_optimal_points(
            model=model,
            bounds=self.std_bounds,
            num_samples=num_pareto_samples,
            num_points=num_pareto_points,
            optimizer=random_search_optimizer,
            optimizer_kwargs=optimizer_kwargs
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)
        # mes_lb = qLowerBoundMultiObjectiveMaxValueEntropySearch(
        #     model=model,
        #     pareto_fronts=pf,
        #     hypercell_bounds=hypercell_bounds,
        #     estimation_type="LB",
        # )
        jes_lb = qLowerBoundMultiObjectiveJointEntropySearch(
            model=model,
            pareto_sets=ps.to(torch.float64),
            pareto_fronts=pf.to(torch.float64),
            hypercell_bounds=hypercell_bounds.to(torch.float64),
            estimation_type="LB",
        )
        # pes = qMultiObjectivePredictiveEntropySearch(model=model, 
        #                                              pareto_sets=ps)
        return jes_lb

    def suggest(self):
        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of `self.n_suggestions` suggestion(s).
                each suggestion is a microarchitecture embedding.
        """
        if self.init:
            '''
                Pure Random Initialization
            '''
            # self.init = False
            # x_init = random.sample(
            #     range(1, self.design_space.size + 1), k=self.n_inits
            # )
            # potential_suggest =  [
            # self.design_space.vec_to_microarchitecture_embedding(
            #     self.design_space.idx_to_vec(_x_guess)
            #     ) for _x_guess in x_init
            # ]
            # return potential_suggest
            
            '''
                Homebrew Latin Hypercube Initialization
            '''
            init_points = latin_hypercube(self.n_inits, self.dim)
            init_points = from_unit_cube(
                init_points, self.lb, self.ub)
            x_suggest = []
            for x in init_points:
                    index = nearest(
                        x, self.lb, self.ub, 
                        self.microarchitecture_embedding_set)
                    x_suggest.append(
                        self.microarchitecture_embedding_set[index].tolist())
            self.init = False
            return x_suggest
            '''
                Botorch sobel draw
            '''
            # x_suggest = []
            # self.init = False
            # init_points = draw_sobol_samples(
            #     self.bounds, self.n_inits, 1, seed=0).reshape([self.n_inits, self.dim]).numpy()
            # for x in init_points:
            #         index = nearest(
            #             x, self.lb, self.ub, 
            #             self.microarchitecture_embedding_set)
            #         x_suggest.append(
            #             self.microarchitecture_embedding_set[index].tolist())
            # return x_suggest
        else:
            print("[BoTorch]: Start to get Acq function ...")
            acq = self.get_ac(self.model)
            
            print("[BoTorch]: Start to evaluate Acq function ...")
            sample_time = 10
            sample_size = 512
            total_acv_val = []
            total_samples = []
            # for now this only works when n_suggest = 1 
            for _ in range(sample_time):
                samples = random.sample(
                    range(self.microarchitecture_embedding_set.shape[0]), 
                    k=sample_size
                )
                X = self.microarchitecture_embedding_set[samples]
                X = normalize(torch.Tensor(X), self.bounds)
                with torch.no_grad():
                    acq_values = acq.forward(X)
                    top_acq_val, indices = torch.topk(acq_values, k=self.n_suggest)
                total_acv_val.append(top_acq_val)
                total_samples.append(samples[indices])
            final_indice = torch.argmax(torch.tensor(total_acv_val).squeeze())
            x_suggest = self.microarchitecture_embedding_set[total_samples[final_indice]]
            if self.n_suggest == 1:
                return [x_suggest.tolist()]
            else:
                return x_suggest.tolist()

    def fit_model(self, train_X, train_Y, num_outputs):
        model = SingleTaskGP(train_X, train_Y, 
                             outcome_transform=Standardize(m=num_outputs))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def observe(self, x, y):
        """
            send an observation of a suggestion back to the optimizer.

            parameters
            ----------
            x: <list> of <list>
                the output of `suggest`.
            y: <list> of <list>
                corresponding values where each `x` is mapped to.
        """
        for _x in x:
            self.x.append(_x)
            idx = np.argwhere(
                np.all(self.microarchitecture_embedding_set == _x, axis=1))
            self.microarchitecture_embedding_set = np.delete(
                self.microarchitecture_embedding_set, idx, axis=0)
        for _y in y:
            _y[1] = -_y[1]
            _y[2] = -_y[2]
            self.y.append(_y)
        train_x = torch.tensor(self.x, dtype=torch.float64)
        train_x = normalize(train_x, self.bounds)
        print("[BoTorch]: Start to fit GP model ...")
        self.model = self.fit_model(train_x,
                                    torch.tensor(self.y, dtype=torch.float64), 
                                    len(self.y[0]))


if __name__ == "__main__":
    experiment(BOTorchOptimizer)
