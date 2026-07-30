"""
thermo.py
Thermodynamique de l'azote (N2) via l'équation d'état cubique de Peng-Robinson,
et polynôme NASA pour les propriétés de gaz parfait.

Traduction directe de (dans l'ordre de dépendance) :
    getThermo.m -> getPfromTandRho.m, getEnergyfromTandRho.m
                -> getTfromPandRho.m, getRhofromTandP.m
                -> getSos.m -> getGammaSandE0S.m

Convention : tous les arguments/tableaux sont des np.ndarray 1D (un point de
grille par élément), on garde donc la vectorisation d'origine.
"""

import numpy as np

# Coefficients du polynôme NASA (7 termes) pour l'azote N2 (Cp/R, gaz parfait)
# Ordre MATLAB an(1)..an(6) -> index Python 0..5 (an(7) n'est jamais utilisé).
NASA_N2 = np.array([
    3.531005280e+00,
    -1.236609870e-04,
    -5.029994370e-07,
    2.435306120e-09,
    -1.408812350e-12,
    -1.046976280e+03,
    2.967474680e+00,
])


def get_thermo(T):
    """getThermo.m — paramètres PR-EOS pour N2 en fonction de T.

    Retourne (a, b, R, dadT, d2adT2). `b` et `R` sont des scalaires,
    les autres ont la forme de T.
    """
    T = np.asarray(T, dtype=float)

    MW = 28.0134e-3
    Tc = 126.19
    pc = 3.3958e+6
    omega = 0.03720

    c = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

    R = 8.314 / MW
    a = 0.457236 * (R * Tc) ** 2 / pc * (1 + c * (1 - np.sqrt(T / Tc))) ** 2
    b = 0.077796 * R * Tc / pc
    G = c * np.sqrt(T / Tc) / (1 + c * (1 - np.sqrt(T / Tc)))
    dadT = -1.0 / T * a * G
    d2adT2 = 0.457236 * R**2 / T / 2 * c * (1 + c) * Tc / pc * np.sqrt(Tc / T)

    return a, b, R, dadT, d2adT2


def get_p_from_T_rho(T, rho):
    """getPfromTandRho.m — pression donnée T et rho (PR-EOS)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)

    a, b, R, dadT, d2adT2 = get_thermo(T)
    v = 1.0 / rho
    p = R * T / (v - b) - a / (v**2 + 2 * v * b - b**2)
    return p


def get_energy_from_T_rho(T, rho):
    """getEnergyfromTandRho.m — énergie interne spécifique donnée T et rho."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    an = NASA_N2

    a, b, R, dadT, d2adT2 = get_thermo(T)
    h_ideal = R * T * (
        an[0] + an[1] * T / 2 + an[2] * T**2 / 3 + an[3] * T**3 / 4
        + an[4] * T**4 / 5 + an[5] / T
    )

    v = 1.0 / rho
    K1 = 1 / np.sqrt(8) / b * np.log((v + (1 - np.sqrt(2)) * b) / (v + (1 + np.sqrt(2)) * b))
    dep = (a - T * dadT) * K1

    e = h_ideal - R * T + dep
    return e


def get_T_from_p_rho(p, rho):
    """getTfromPandRho.m — inversion de Newton : température donnée p et rho."""
    p = np.asarray(p, dtype=float)
    rho = np.asarray(rho, dtype=float)

    CRIT = 1.0e-6

    v = 1.0 / rho
    T = np.full_like(rho, 300.0)  # valeur initiale, comme en MATLAB
    p_n = get_p_from_T_rho(T, rho)
    diff = p_n - p

    while np.max(np.abs(diff)) > CRIT:
        a, b, R, dadT, d2adT2 = get_thermo(T)
        dpdT = R / (v - b) - 1.0 / (v**2 + 2 * v * b - b**2) * dadT
        T = T - diff / dpdT
        p_n = get_p_from_T_rho(T, rho)
        diff = p_n - p

    return T


def get_rho_from_T_p(T, p):
    """getRhofromTandP.m — racine cubique analytique (Cardano) : rho donné T et p.

    NB : la formule sign(x)*|x|^(1/3) est utilisée volontairement (comme en
    MATLAB) pour éviter que numpy ne retourne une racine complexe en élevant
    un nombre négatif à une puissance fractionnaire.
    """
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)

    a, b, R, dadT, d2adT2 = get_thermo(T)

    a0 = p * b**3 + b**2 * R * T - a * b
    a1 = -3 * p * b**2 - 2 * b * R * T + a
    a2 = p * b - R * T
    a3 = p

    # une seule racine réelle attendue dans la plage physique
    l = a2 / a3
    m = a1 / a3
    n = a0 / a3
    A = 1.0 / 3 * (3 * m - l**2)
    B = 1.0 / 27 * (2 * l**3 - 9 * l * m + 27 * n)
    D = A**3 / 27 + B**2 / 4

    def signed_cbrt(x):
        return np.sign(x) * np.abs(x) ** (1.0 / 3)

    M = signed_cbrt(-B / 2 + np.sqrt(D))
    N = signed_cbrt(-B / 2 - np.sqrt(D))

    v = M + N - l / 3
    rho = 1.0 / v
    return rho


def get_sos(V):
    """getSos.m — vitesse du son donnée les variables primitives V = [rho; u; p]."""
    an = NASA_N2

    rho = V[0, :]
    p = V[2, :]
    v = 1.0 / rho

    T = get_T_from_p_rho(p, rho)
    a, b, R, dadT, d2adT2 = get_thermo(T)

    cp_ideal = R * (an[0] + an[1] * T + an[2] * T**2 + an[3] * T**3 + an[4] * T**4)
    cv_ideal = cp_ideal - R

    dpdT = R / (v - b) - dadT / (v**2 + 2 * v * b - b**2)
    dpdv = -R * T / (v - b) ** 2 * (
        1 - 2 * a * ((R * T * (v + b) * ((v**2 + 2 * v * b - b**2) / (v**2 - b**2)) ** 2) ** (-1))
    )
    K1 = 1 / np.sqrt(8) / b * np.log((v + (1 - np.sqrt(2)) * b) / (v + (1 + np.sqrt(2)) * b))
    cp = cv_ideal - T * dpdT**2 / dpdv - K1 * T * d2adT2

    kT = -1.0 / v / dpdv
    av = -dpdT / v / dpdv
    ks = kT - v * T * av**2 / cp

    sos = np.sqrt(1.0 / rho / ks)
    return sos


def get_gammaS_e0S(V):
    """getGammaSandE0S.m — gamma effectif et énergie de référence pour le modèle double-flux."""
    rho = V[0, :]
    p = V[2, :]

    T = get_T_from_p_rho(p, rho)
    e = get_energy_from_T_rho(T, rho)
    sos = get_sos(V)

    gammaS = sos**2 * rho / p          # Eq. (12)
    e0S = e - p / rho / (gammaS - 1)   # Eq. (16)
    return gammaS, e0S


# ---------------------------------------------------------------------------
# Ajouts ci-dessous : PAS dans le MATLAB d'origine. Necessaires uniquement
# pour le schema "fully conservative" (FC) de comparaison (Section 4.1 /
# Fig. 4a du papier Ma, Lv & Ihme 2017), qui doit inverser T a partir de
# (e, rho) -- alors que le modele double-flux n'a jamais besoin de cette
# inversion (il inverse seulement (p, rho) -> T, deja fourni par
# get_T_from_p_rho). Cf. Eq. (43) du papier pour cv.
# ---------------------------------------------------------------------------

def get_cv_from_T_rho(T, rho):
    """Chaleur specifique a volume constant, cv = (de/dT)|rho (Eq. 43 du papier).

    cv = cv_ideal - K1*T*d2a/dT2, avec cv_ideal = cp_ideal - R (polynome NASA).
    """
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    an = NASA_N2

    a, b, R, dadT, d2adT2 = get_thermo(T)
    cp_ideal = R * (an[0] + an[1] * T + an[2] * T**2 + an[3] * T**3 + an[4] * T**4)
    cv_ideal = cp_ideal - R

    v = 1.0 / rho
    K1 = 1 / np.sqrt(8) / b * np.log((v + (1 - np.sqrt(2)) * b) / (v + (1 + np.sqrt(2)) * b))

    cv = cv_ideal - K1 * T * d2adT2
    return cv


def get_T_from_e_rho(e, rho, T_init=300.0, max_iter=100, rtol=1e-10):
    """Inversion de Newton : temperature donnee energie interne specifique et rho.

    Utilise uniquement par le schema fully-conservative (FC), pour l'etape
    "iteratively solve for p or T, given rho, e" du flowchart Fig. 4(a) du
    papier. Le modele double-flux evite entierement cette iteration grace a
    Eq. (22) (pression directe) + get_T_from_p_rho (inversion sur p, pas e).

    Contrairement a une premiere version, cette fonction :
    - valide rho (fini, positif) AVANT tout appel a la thermodynamique ;
    - amortit le pas de Newton (evite un T qui explose si cv est petit) ;
    - leve explicitement RuntimeError en cas de non-convergence, plutot que
      de retourner silencieusement un T non converge apres max_iter.
    """
    e = np.asarray(e, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError(
            f"Densite invalide passee a get_T_from_e_rho : "
            f"min(rho)={np.min(rho) if rho.size else float('nan')}"
        )
    if np.any(~np.isfinite(e)):
        raise ValueError("Energie interne non-finie passee a get_T_from_e_rho.")

    T = np.full_like(rho, T_init)
    e_n = get_energy_from_T_rho(T, rho)
    diff = e_n - e

    scale = np.maximum(np.abs(e), 1.0)  # tolerance relative (e ~ 1e5-1e6 J/kg)
    it = 0
    while np.max(np.abs(diff) / scale) > rtol and it < max_iter:
        cv = get_cv_from_T_rho(T, rho)

        if np.any(~np.isfinite(cv)):
            raise RuntimeError("cv non fini durant l'inversion T(e,rho).")
        if np.any(cv <= 0.0):
            raise RuntimeError(
                f"Etat thermodynamiquement invalide durant l'inversion T(e,rho) : "
                f"min(cv)={np.min(cv):.6e} (cv doit etre strictement positif)"
            )
        cv_safe = np.maximum(cv, 1.0e-6)

        dT = -diff / cv_safe
        dT = np.clip(dT, -0.25 * T, 0.25 * T)  # amortissement : max +/-25% de T par iteration
        T = T + dT
        T = np.maximum(T, 1.0)  # garde-fou : T doit rester positive
        e_n = get_energy_from_T_rho(T, rho)
        diff = e_n - e
        it += 1

    residual = np.max(np.abs(diff) / scale)
    if residual > rtol:
        raise RuntimeError(
            f"Inversion T(e,rho) non convergee apres {max_iter} iterations : "
            f"residu relatif max = {residual:.3e} (tolerance = {rtol:.1e})"
        )

    return T
