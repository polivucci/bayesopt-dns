import numpy as np

from emukit.core.loop import UserFunctionResult
from tools import read_examples,random_exploration,function_evaluation,bo_model
from configuration import *

if __name__=="__main__":

    # Initialise external simulation evaluation:
    run_simulation = function_evaluation(external_solver_interface, 
                                         X_transform = X_transform, 
                                         Y_transform = Y_transform)

    # Generate initial training data:
    if resume:
        # Read training data from file:
        X, Y_cost, Y_constraint = read_examples(training_data_file)
    else:
        # Generate training data from exploration of parameter space:
        X, Y_cost, Y_constraint = random_exploration(space, 
                                                     n_initial_data_points, 
                                                     run_simulation)
    
    # Initialise GP model on the training data:
    bo = bo_model(X, Y_cost, Y_constraint, space)
    n_points_ini = bo.loop_state.X.shape[0]

    results = None
    print('Running optimization for '+str(n_iterations)+' iterations ... \n')
    for n in range(n_iterations):

        # Generate next evaluation point:
        X_new = bo.get_next_points(results)

        # Evaluate parameters:
        Y_cost_new, Y_constraint_new = run_simulation(X_new)

        # Update training data with new evaluation:
        results=[UserFunctionResult(X_new[0], Y_cost_new[0], Y_constraint=Y_constraint_new[0])]
        
        print('Iteration '+str(n+1)+', X=', X_new, 'cost=', Y_cost_new, 'constr', Y_constraint_new)

    X = bo.loop_state.X
    Y_cost = bo.loop_state.Y
    Y_constraint = bo.loop_state.Y_constraint

    # Check we got the correct number of points
    print(bo.loop_state.X.shape[0])
    print(n_iterations)
    print(n_points_ini)
    assert bo.loop_state.X.shape[0] == n_iterations + n_points_ini

    # Export model to 3D VTK files for visualisation:
    model_to_vtk(bo)