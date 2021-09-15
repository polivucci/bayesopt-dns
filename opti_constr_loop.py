import numpy as np

from optimization.prm import *
from tools import read_examples,random_exploration,function_evaluation
from emukit.core.loop import UserFunctionResult
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.constraints import LinearInequalityConstraint

if __name__=="__main__":

    if resume:
        X, Y_cost, Y_constraint = read_examples(training_data_file)
    else:
        X, Y_cost, Y_constraint = random_exploration(space, initial_data_points)
    
    bo = bo_model(X, Y_cost, Y_constraint)

    results = None
    print('Running optimization for '+str(num_iterations)+' iterations ... \n')
    for n in range(num_iterations):

        X_new = bo.get_next_points(results)

        Y_cost_new, Y_constraint_new = function_evaluation(X_new, 
                                                           X_transform=X_transform,
                                                           Y_transform=Y_transform)

        results=[UserFunctionResult(X_new[0], Y_cost_new[0], Y_constraint=Y_constraint_new[0])]
        
        print('Iteration '+str(n+1)+', X=', X_new, 'cost=', Y_new, 'constr', Yc_new)

    X = bo.loop_state.X
    Y_cost = bo.loop_state.Y
    Y_constraint = bo.loop_state.Y_constraint

    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations + initial_data_points

    # Export model to 3D VTK files for visualisation:
    model_to_vtk(bo)