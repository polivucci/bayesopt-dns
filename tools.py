import numpy as np
from dns import dns

def function_evaluation(X: np.ndarray, X_transform=None, Y_transform=None) -> tuple:
    '''!
    Wrapper around the external evaluation of the cost function.

    @param X: the evaluation coordinates.
    @param X_transform: optional parameter transformation.
    @param Y_transform: optional output transformation.

    @return: tuple containing the objective and the constraint value.
    '''

    if X_transform is not None:
        # Inverse-transform data to get physical values for simulaton input:
        X = X_transform(X)

    # Evaluate DNS simulation:
    Y_cost, Y_constraint = dns(X)
    
    if Y_transform is not None:
        # Transform prediction value:
        (Y_cost, Y_constraint) = Y_transform((Y_cost, Y_constraint))

    return Y_cost, Y_constraint


def random_exploration(space: 'ParameterSpace', initial_data_points: int):
    '''!
    Parameter-space exploration via Latin hypercube sampling.
    '''
    
    print('Evaluating '+str(initial_data_points)+' intial designs ... ')
    from emukit.experimental_design.model_free.latin_design import LatinDesign
    design = LatinDesign(space) 
    X = design.get_samples(initial_data_points)
    
    Y_cost, Y_constraint = function_evaluation(X, 
                                               X_transform=X_transform,
                                               Y_transform=Y_transform)

    return X, Y_cost, Y_constraint

def read_examples(csv_file: str):
    '''
    Read in examples from data file.
    '''
    print('Reading in training data ... ')
    Xx, Xy, Xz, Y_cost, W, Y_constraint = np.genfromtxt(csv_file, unpack=True)
    X=np.array([Xx, Xy, Xz]).transpose()

    return X, Y_cost, Y_constraint


def bo_model(X, Y_cost, Y_constraint):
    '''
    Builds GP model, acquisition function and sets up BO loop based on input data.
    '''

    import GPy
    from emukit.model_wrappers import GPyModelWrapper
    
    # Cost function GP model definition:
    kern_obj = GPy.kern.Matern52(len(space.parameters), variance=1., ARD=True)
    model_gpy_mcmc = GPy.models.GPRegression(X, Y_cost, kernel=kern_obj, noise_var=0.001)
    model_gpy_mcmc.kern.set_prior(GPy.priors.Uniform(0,5))
    model_obj = GPyModelWrapper(model_gpy_mcmc)

    # Constraint function GP model definition:
    kern_constr = GPy.kern.Matern52(len(space.parameters), variance=1., ARD=True)
    model_gpy_mcmc_constr = GPy.models.GPRegression(X,Y_constraint,kernel=kern_constr,noise_var=0.001)
    model_gpy_mcmc_constr.kern.set_prior(GPy.priors.Uniform(0,5))
    model_constr = GPyModelWrapper(model_gpy_mcmc_constr)

    # Calculating the constrained acquisition function:
    from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, ProbabilityOfFeasibility
    from emukit.core.acquisition import IntegratedHyperParameterAcquisition
    # Unconstrained acquisition:
    ei = IntegratedHyperParameterAcquisition(model_obj, ExpectedImprovement)
    # Probability of feasibility:
    pof = IntegratedHyperParameterAcquisition(model_constr, ProbabilityOfFeasibility)
    # Putting it together:
    acquisition_constrained = ei * pof

    from emukit.bayesian_optimization.loops import UnknownConstraintBayesianOptimizationLoop
    # Optimizer setup :
    bo = UnknownConstraintBayesianOptimizationLoop(model_objective = model_obj,
                                                   model_constraint = model_constr,
                                                   space = space,
                                                   acquisition = acquisition_constrained,
                                                   batch_size = 1)

    return bo