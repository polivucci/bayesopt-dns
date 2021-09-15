import numpy as np

class data_transform():
  '''!
  Base transform class.
  '''
  def __init__(self, *args, **kwargs):

  def fwd(self, X):
      '''
    Forward transform.
    '''
    pass

  def bwd(self, X):
    '''
    Inverse transform.
    '''
    pass


class compound_transform(data_transform):
  '''!
  Compounded transformation.
  '''
  def __init__(self, transform1: 'data_transform', transform2: 'data_transform'):
    self.transform1 = transform1
    self.transform2 = transform2

  def fwd(self, X):
    return transform2.fwd(transform1.fwd(X))

  def bwd(self, X):
    return transform1.bwd(transform2.bwd(X))


class normalize(data_transform):
  '''!
  Normalises data range.
  '''
  def __init__(self, normalize_constant: tuple):
    self.normalize_constant = np.array(normalize_constant)

  def fwd(self, Y):
    return Y/self.normalize_constant

  def bwd(self, Y):
    return Y*self.normalize_constant


class quadratic(data_transform):
  '''!
  Quadratic transformation for Tm.
  '''
  def __init__(self, quadratic_constants: tuple):


  def fwd(self, X):
    X_out = np.copy(X)
    X_out[:,2] = X[:,2]*(self.a[0]*X[:,0]*X[:,0]+self.a[1]*X[:,0]+self.a[2])
    return X_out

  def bwd(self, X):
    X_out = np.copy(X)
    X_out[:,2] = X[:,2]/(self.a[0]*X[:,0]*X[:,0]+self.a[1]*X[:,0]+self.a[2])
    return X_out
