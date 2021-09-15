import numpy as np

def model_upto(bo, n):

  print('check')

  Xx, Xy, Xz, P, W, G = np.genfromtxt("inpdata-new.dat",unpack=True)

  X=np.array([Xx, Xy, Xz]).transpose()[:n,:]
  Y_cost=-G[:n,None]
  Y_constraint=P[:n,None]-0.966

  bo = bo_model(X, Y_cost, Y_constraint)

  maxloc, maxei = bo.acquisition_optimizer.optimize(bo.acquisition)

  return maxloc, maxei

for n in range(36,44):
  maxloc, ei_max = model_upto(n)
  print (ei_max)
  with open('replay_EI.dat', 'a') as f:
    f.write(('%4.4f'+'\n') % (ei_max))

def model_to_vtk(bo: 'OuterLoop') -> None:
  '''!
  Export the model to 3D structured VTK files for visualisation.

  @param bo: emukit Bayesian optimisation loop object.
  '''

  # generate a discrete grid in parameter space:
  p = bo.space.parameters
  ax = list(map(list, zip(*[[p[i].min, p[i].max] for i in range(3)])))
  x = np.linspace(ax[0], ax[1], 50)
  xx, yy, zz = np.meshgrid(*x.T, indexing='ij')
  grid = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).T
  
  # compute model at grid points:
  model = {}
  model['mu'], model['var'] = bo.model_objective.predict(grid)
  model['sig']  = np.sqrt(var[:,0])
  model['acq']  = bo.acquisition.evaluate(grid)
  model['ei']   = bo.ei.evaluate(grid)
  model['pof']  = bo.pof.evaluate(grid)

  from pyevtk.hl import gridToVTK
  
  # export to VTK
  for element in model:
    field = np.reshape(model[element], xx.shape)
    gridToVTK(str(element), x[0], x[1], x[2], pointData = {str(element): field})
  
  return None