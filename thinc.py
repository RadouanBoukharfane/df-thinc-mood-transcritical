"""
thinc.py
Reconstruction THINC (Tangent of Hyperbola for INterface Capturing)
scalaire, conservative en moyenne cellulaire, sur maillage uniforme
periodique.

Cette premiere brique est volontairement ISOLEE : elle ne contient ni
selection BVD, ni detecteur contact/choc, ni couplage au double-flux.

Convention des sorties
-----------------------
Pour chaque cellule i :

    q_left[..., i]  = valeur issue de la cellule i a sa face gauche i-1/2
    q_right[..., i] = valeur issue de la cellule i a sa face droite i+1/2

Ainsi, a la face i+1/2, les deux etats candidats sont :

    q_L_face[..., i] = q_right[..., i]
    q_R_face[..., i] = q_left[..., i+1]

Le profil THINC dans la cellule i est

    q_i(xi) = q_min + 0.5 * dq *
              [1 + theta * tanh(beta * (xi - xi0))],

avec xi dans [0, 1]. La position xi0 est determinee de facon que la
moyenne du profil soit exactement la moyenne cellulaire q_i.

Activation
----------
THINC est active uniquement dans une cellule localement monotone :

    (q_i - q_{i-1}) * (q_{i+1} - q_i) > 0.

Sinon, ou lorsque l'amplitude locale est trop petite, on retourne le
profil constant q_i aux deux faces. Cette branche est indispensable pour
les plateaux, extrema lisses et divisions presque singulieres.
"""

import numpy as np


def _validate_beta(beta):
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta doit etre fini et strictement positif, obtenu {beta!r}")
    return beta


def reconstruct_thinc(V, beta=1.6, eps=1.0e-14):
    """Reconstruit les valeurs THINC gauche/droite de chaque cellule.

    Parameters
    ----------
    V : array_like
        Tableau de forme ``(Nx,)`` ou ``(..., Nx)``. L'axe spatial est le
        dernier axe. Les conditions aux limites sont periodiques.
    beta : float, optional
        Parametre de raideur THINC. Une valeur autour de 1.5--2.0 est un
        choix courant pour un premier prototype. Par defaut : 1.6.
    eps : float, optional
        Tolerance relative utilisee pour detecter une amplitude locale
        quasi nulle et pour proteger les logarithmes. Par defaut : 1e-14.

    Returns
    -------
    q_left, q_right : ndarray
        Tableaux de meme forme que ``V`` : valeurs issues de chaque cellule
        a sa face gauche et a sa face droite.

    Notes
    -----
    Dans les cellules THINC actives, le profil est borne entre les deux
    valeurs voisines ``V[i-1]`` et ``V[i+1]``. Dans les cellules inactives,
    les deux valeurs de face sont exactement ``V[i]``.
    """
    q = np.asarray(V, dtype=float)
    if q.ndim == 0:
        raise ValueError("V doit posseder un axe spatial")
    if q.shape[-1] < 3:
        raise ValueError("THINC requiert au moins 3 cellules periodiques")

    beta = _validate_beta(beta)
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps doit etre fini et strictement positif, obtenu {eps!r}")

    axis = -1
    q_im1 = np.roll(q, shift=1, axis=axis)
    q_ip1 = np.roll(q, shift=-1, axis=axis)

    q_min = np.minimum(q_im1, q_ip1)
    q_max = np.maximum(q_im1, q_ip1)
    dq = q_max - q_min

    # Cellule strictement monotone : q_i se trouve entre ses deux voisins.
    monotone = (q - q_im1) * (q_ip1 - q) > 0.0

    # Tolerance locale homogene en unite : evite de desactiver abusivement
    # THINC lorsque q est tres grand, tout en traitant exactement les plateaux.
    local_scale = np.maximum.reduce(
        [np.ones_like(q), np.abs(q_im1), np.abs(q), np.abs(q_ip1)]
    )
    nondegenerate = dq > eps * local_scale
    active = monotone & nondegenerate

    # Valeur de repli : reconstruction constante, donc preservant exactement
    # les constantes et les extrema locaux.
    q_left = q.copy()
    q_right = q.copy()

    if not np.any(active):
        return q_left, q_right

    theta = np.sign(q_ip1 - q_im1)

    # Moyenne normalisee C dans [0, 1]. Le clipping n'agit que comme garde
    # numerique ; pour une cellule monotone exacte, C appartient deja a (0,1).
    C = np.zeros_like(q)
    np.divide(q - q_min, dq, out=C, where=nondegenerate)
    tiny = np.finfo(float).eps
    C = np.clip(C, tiny, 1.0 - tiny)

    # Si h(xi)=tanh(beta*(xi-xi0)), la contrainte de moyenne impose
    # h_bar = theta*(2*C-1). L'equation peut etre inversee explicitement :
    #
    #   y = exp(2 beta xi0)
    #     = [exp(beta)-r] / [r-exp(-beta)],
    #   r = exp(beta h_bar).
    #
    # On limite h_bar a l'intervalle ouvert (-1,1) afin de proteger les cas
    # tres proches des bornes en precision finie.
    h_bar = theta * (2.0 * C - 1.0)
    h_bar = np.clip(h_bar, -1.0 + tiny, 1.0 - tiny)

    exp_beta = np.exp(beta)
    exp_minus_beta = np.exp(-beta)
    r = np.exp(beta * h_bar)

    numerator = exp_beta - r
    denominator = r - exp_minus_beta

    # Les quantites sont strictement positives en theorie. Le maximum protege
    # uniquement contre un arrondi sous-flottant exceptionnel.
    positive_floor = np.finfo(float).tiny
    y = np.maximum(numerator, positive_floor) / np.maximum(denominator, positive_floor)
    xi0 = 0.5 * np.log(y) / beta

    h_left = np.tanh(-beta * xi0)          # xi = 0
    h_right = np.tanh(beta * (1.0 - xi0)) # xi = 1

    q_left_thinc = q_min + 0.5 * dq * (1.0 + theta * h_left)
    q_right_thinc = q_min + 0.5 * dq * (1.0 + theta * h_right)

    q_left = np.where(active, q_left_thinc, q_left)
    q_right = np.where(active, q_right_thinc, q_right)

    # Ultime protection contre un depassement de quelques ulp : dans une
    # cellule active, le profil mathematique appartient a [q_min, q_max].
    q_left = np.where(active, np.clip(q_left, q_min, q_max), q_left)
    q_right = np.where(active, np.clip(q_right, q_min, q_max), q_right)

    return q_left, q_right


def reconstruct_thinc_left(V, beta=1.6, eps=1.0e-14):
    """Valeur issue de chaque cellule a sa face gauche i-1/2."""
    q_left, _ = reconstruct_thinc(V, beta=beta, eps=eps)
    return q_left


def reconstruct_thinc_right(V, beta=1.6, eps=1.0e-14):
    """Valeur issue de chaque cellule a sa face droite i+1/2."""
    _, q_right = reconstruct_thinc(V, beta=beta, eps=eps)
    return q_right
