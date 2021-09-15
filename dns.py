import numpy as np
import os
from pproc import pproc_simulation

def dns(X) -> np.ndarray:
  '''!
  Interface with the external DNS solver.

  @param X: the control parameter values.

  @return: a ndarray containing the objective and the constraint value.
  '''

  global ncase

  X = X.reshape((len(X),3))
  X_out = np.copy( X.reshape((len(X),3)) )
  n = X.shape[0]

  cost=np.empty(n,float)
  constraint=np.empty(n,float)

  # Loop to evaluate n different parameter combinations 
  # (useful for initial exploration):
  for j in range(0,n):
    try:
      ncase+=1
    except:
      ncase=1

    name_case='c'+str(ncase).zfill(3)+"/"
    path_case=path_solver+name_case
    os.system("cp -r "+path_solver+"basecase/ "+path_case) #create new case

    # Write parameters into incompact3d.prm:
    with open(path_case+"BC-Channel-flow.prm", "r") as f:
      prm = f.readlines()
    prm[39]=str(X[j,2])+'  #t_stall\n'
    prm[40]=str(X[j,0])+'  #w_max/wss\n'
    prm[41]=str(X[j,1])+'  #w_thresh/wss\n'
    with open(path_case+"BC-Channel-flow.prm", 'w') as f:
      for line in prm:
        f.write(line)

    try:
      # Submit job:
      print('Submitting job '+str(ncase)+' ... ')
      # os.system("cd "+path_case+"; mpirun -np 2 ./incompact3d > log.inc3d")
      os.system("cd "+path_case+"; qsub -Wblock=true -N c"+str(ncase).zfill(3)+" run_job.pbs")
      try:
      #post-processing to evaluate objective:
      pproc = pproc_simulation( D=D, pathin=path_case )
      cost[j]=pproc["cost"]
      constraint[j]=pproc["constraint"]
      except:
        cost[j]=float('nan')
        constraint[j]=float('nan')
        print('Design '+str(ncase)+': pproc not successful !')
    except:
      cost[j]=float('nan')
      constraint[j]=float('nan')
      print('DNS '+str(ncase)+' not successful !')

    with open("output.dat", 'a') as f:
      if (ncase==1): f.write( ("%s\t\t"*6+"\n") % ('# case','x','y','z','obj','w/wss') )
      f.write(("%0.3i\t"+"%6.6f\t"*5+"\n") % (
                    ncase, X_out[j,0], X_out[j,1], X_out[j,2], cost[j], constraint[j]) )

  out = np.array([cost.reshape(n,1)[:,0], constraint.reshape(n,1)[:,0]])
  return out[:,0,None], out[:,1,None]


def pproc_simulation( D: float=None, n_discs: int=16, pathin='./' ) -> dict:
  '''!
  This function processes the raw simulation output and outputs cost and constraint.

  @param D
  @param n_discs

  @return pproc: dictionary containing post-processing output.
  '''
  power_avg=0.;  power_pump_avg=0.;  power_disc_avg=0.; w_mean=0.

  dudy = np.loadtxt(pathin+'raw-data/wall_grad.dat',usecols=(1,))
  t = np.loadtxt(pathin+'raw-data/disc_04.dat',usecols=(0,))
  
  cf = dudy/(0.5*Re*Ub*Ub)    # skin-friction coefficient
  pwrp = 0.5*D*D*cf*Ub*Ub*Ub  # pumping power per flow unit
  P0 = 0.5*p0*D*D             # reference case total power on per flow unit

  for j in range(1,n_discs/2+1):
    for a in ('','_up'):
      tm=np.loadtxt(pathin+'raw-data/disc'+a+'_0'+str(j)+'.dat',usecols=(1,))
      omg=np.loadtxt(pathin+'raw-data/disc'+a+'_0'+str(j)+'.dat',usecols=(3,))

      w_mean = w_mean+np.mean(np.abs(omg)*D*0.5)

      omg_dot = np.gradient(omg, t) #edge_order=1)
      mask=(tm==0.)
      omg[mask]=0.
      omg_dot[mask]=0.

      pwrd=np.abs(tm*omg)
      pwr=pwrp+pwrd

      power_disc_avg=power_disc_avg+np.mean(pwrd)

  power_pump_avg=np.mean(pwrp[n_ini:])
  power_disc_avg=power_disc_avg/n_discs
  w_mean=w_mean/n_discs
  power_avg=power_pump_avg+power_disc_avg

  pproc = {
  "cost"            : power_avg/P0,                       # this is tot pwr
  "constraint"      : (1-power_pump_avg)/power_disc_avg,  # this is G
  "power_avg"       : power_avg/P0,
  "power_pump_avg"  : power_pump_avg/P0,
  "power_disc_avg"  : power_disc_avg/P0,
  "w_mean"          : w_mean/W
  }

  return pproc