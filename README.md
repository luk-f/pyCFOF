[![PyPI](https://github.com/luk-f/pyCFOF/actions/workflows/python-publish.yml/badge.svg)](https://github.com/luk-f/pyCFOF/actions/workflows/python-publish.yml)

# pyCFOF

## Pour commencer

### Installation

Lancer `pip install -r requirements.txt` ou `python3 -m pip install -r requirements.txt`.

Ou à partir du dépôt `pip install Concentration-Free-Outlier-Factor`.

### Utilisation

    >>> from pyCFOF import ConcentrationFreeOutlierFactor as CFOF
    >>> X = [[-1.1], [0.2], [101.1], [0.3]]
    >>> cfof = CFOF(n_neighbors=len(X), rho=[0.1])
    >>> cfof.fit_predict(X)
    array([[ 1],
           [ 1],
           [-1],
           [ 1]])
    >>> cfof.outlier_factor_
    array([[0.75],
           [0.5 ],
           [1.  ],
           [0.5 ]])

## Remerciements

Développements des travaux de :
 - Fabrizio Angiulli, [CFOF: A Concentration Free Measure for Anomaly Detection. ACM Transactions on Knowledge Discovery from Data (TKDD), 14(1):Article 4, 2020](https://dl.acm.org/doi/abs/10.1145/3362158)


Utilisation de :
 - [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](https://scikit-learn.org/stable/index.html)
