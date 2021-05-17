import numpy as np
from _cfof import ConcentrationFreeOutlierFactor as CFOF

import matplotlib.pyplot as plt

dataset = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=(90, 2)), 
                          np.random.normal(loc=60.0, scale=1.0, size=(10, 2))), axis=0)

print(dataset.shape)

rho_list = [0.1, 0.5]
cfof = CFOF(n_neighbors=len(dataset), rho=rho_list)

_ = cfof.fit_predict(dataset)

print(cfof.outlier_factor_)

plt.figure(figsize=(5, 4))

plt.grid()

rho_ite = 0

plt.title(f"rho = {rho_list[rho_ite]}")

# plt.subplots(121)
plt.scatter(dataset[:,0], dataset[:,1], c=cfof.outlier_factor_[:,rho_ite])

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)

# plt.subplot(122)
# plt.scatter(dataset[:,0], dataset[:,1], c=cfof.outlier_factor_[:,1])

_ = plt.show()