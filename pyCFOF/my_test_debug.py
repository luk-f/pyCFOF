from _cfof import ConcentrationFreeOutlierFactor as CFOF

import numpy as np 

dataset = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=(9_000, 2)), 
                          np.random.normal(loc=6.0, scale=1.0, size=(1_000, 2))), axis=0)

rho_list = [0.2, 0.99]
cfof = CFOF(n_neighbors=len(dataset), rho=rho_list)

_ = cfof.fit_predict(dataset)