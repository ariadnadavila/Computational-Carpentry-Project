from __future__ import annotations
import re
import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce


element_re = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(formula: str) -> dict:
    counts = {}
    for (elem, num) in element_re.findall(formula):
        n = int(num) if num else 1
        counts[elem] = counts.get(elem, 0) + n
    if not counts:
        raise ValueError(f"Could not parse formula: {formula}")
    return counts

def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_many(ints):
    return reduce(lcm, ints, 1)

def _elements_by_side(reactants, products):
    rset = set()
    pset = set()
    for f in reactants:
        rset.update(parse_formula(f).keys())
    for f in products:
        pset.update(parse_formula(f).keys())
    return rset, pset

def balance_reaction(reactants, products, tol=1e-10):
    # Early impossibility: elements only on one side
    rset, pset = _elements_by_side(reactants, products)
    only_in_reactants = sorted(rset - pset)
    only_in_products = sorted(pset - rset)
    if only_in_reactants or only_in_products:
        parts = []
        if only_in_reactants:
            parts.append(f"present only in reactants: {', '.join(only_in_reactants)}")
        if only_in_products:
            parts.append(f"present only in products: {', '.join(only_in_products)}")
        raise ValueError("Reaction not possible with the chosen species because some elements are missing on one side ("
                         + "; ".join(parts) + ").")
    
    species = reactants + products
    parsed = [parse_formula(f) for f in species]
    elements = sorted({e for m in parsed for e in m.keys()})
    
    A = np.zeros((len(elements), len(species)), dtype=float)
    for i, el in enumerate(elements):
        for j, m in enumerate(parsed):
            count = m.get(el, 0)
            if j >= len(reactants):  # product side negative
                count = -count
            A[i, j] = count
    
    # SVD nullspace
    U, S, Vt = np.linalg.svd(A)
    v = Vt[-1, :]
    v[np.abs(v) < tol] = 0.0
    if np.allclose(v, 0, atol=tol):
        raise RuntimeError("Nullspace vector is numerically zero; try different tolerance.")
    
    # Normalize to minimal integers
    nonzero = v[np.abs(v) > tol]
    v_scaled = v / np.min(np.abs(nonzero))
    fracs = [Fraction(float(x)).limit_denominator(1000) for x in v_scaled]
    denoms = [f.denominator for f in fracs]
    L = lcm_many(denoms)
    ints = np.array([int(f.numerator * (L // f.denominator)) for f in fracs], dtype=int)
    
    # Flip sign to make most entries positive if needed
    if np.sum(ints < 0) > np.sum(ints > 0):
        ints *= -1
    
    # Reduce GCD
    nonzero_ints = ints[ints != 0]
    if len(nonzero_ints) == 0:
        raise ValueError("Reaction not possible with the chosen species (no non-zero solution).")
    g = reduce(gcd, map(int, np.abs(nonzero_ints)))
    if g > 1:
        ints //= g
    
    # Feasibility: require all strictly positive coefficients for chosen species
    if np.any(ints <= 0):
        raise ValueError("Reaction not possible since products are not registered as products of the reactants.")
    
    return ints.tolist()

def format_equation(reactants, products, coeffs):
    lhs = " + ".join([f"{c if c!=1 else ''}{r}".strip() for c, r in zip(coeffs[:len(reactants)], reactants)])
    rhs = " + ".join([f"{c if c!=1 else ''}{p}".strip() for c, p in zip(coeffs[len(reactants):], products)])
    return f"{lhs} → {rhs}"

# --- Predefined "menus" ---
all_reactants = ["H2", "O2", "C3H8", "Fe", "N2", "HCl"]
all_products = ["H2O", "CO2", "Fe2O3", "NH3", "Cl2"]

def check_species(reactants, products):
    bad_reactants = [r for r in reactants if r not in all_reactants]
    bad_products  = [p for p in products if p not in all_products]
    overlap = set(reactants) & set(products)
    if bad_reactants or bad_products:
        msg = []
        if bad_reactants:
            msg.append(f"Invalid reactants: {', '.join(bad_reactants)} "
                       f"(must be chosen from {all_reactants})")
        if bad_products:
            msg.append(f"Invalid products: {', '.join(bad_products)} "
                       f"(must be chosen from {all_products})")
        if overlap:
            msg.append(f"Invalid overlap: {', '.join(overlap)} appears as both reactant and product")
        raise ValueError("Reaction not possible: " + " | ".join(msg))

def balance_with_output(reactants, products):
    """
    Validates menus, balances, and returns:
      - coeffs: list[int] in order [reactants..., products...]
      - equation: balanced reaction string
    Raises ValueError with a clear message if invalid or impossible.
    """
    check_species(reactants, products)
    coeffs = balance_reaction(reactants, products)
    eq = format_equation(reactants, products, coeffs)
    return coeffs, eq

def try_balance(reactants, products):
    try:
        coeffs, eq = balance_with_output(reactants, products)
        print("Balanced:", eq)
        print("Coefficients:", coeffs)
        return coeffs, eq
    except Exception as e:
        print("Error:", str(e))
        return None, None

# Valid
try_balance(["C3H8", "O2"], ["CO2", "H2O"])
try_balance(["Fe", "O2"], ["Fe2O3"])
try_balance(["N2", "H2"], ["NH3"])

# Impossible 1: element (N) only in products
try_balance(["H2", "O2"], ["NH3"])

# Impossible 2: wrong pairing despite same elements (e.g., H2 + O2 -> H2 + O2)
try_balance(["H2", "O2"], ["H2", "O2"])



