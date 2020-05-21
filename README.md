# CFOF

## Pour commencer

### Installation

Lancer `pip install -r requirements.txt` ou `python3 -m pip install -r requirements.txt`.

### Utilisation

    >>> from pyCFOF import ConcentrationFreeOutlierFactor as CFOF
    >>> X = [[-1.1], [0.2], [101.1], [0.3]]
    >>> cfof = CFOF(n_neighbors=len(X), rho=0.1)
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
 - [CFOF: A Concentration Free Measure for Anomaly Detection, par Fabrizio Angiulli](https://dl.acm.org/doi/abs/10.1145/3362158)
