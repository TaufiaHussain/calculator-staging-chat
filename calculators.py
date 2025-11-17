"""
Backend calculator functions for the Versatile Lab Solution Calculator.

Each function here is a pure calculator:
- Takes numeric / string parameters.
- Returns a dict with results (ready for logging, JSON, or LLM explanation).

Used by:
- Tier-5 chat (via llm_router + OpenAI tools)
- Batch CSV calculator
"""

from __future__ import annotations
from typing import Dict, Any, Callable, List
import math

# ------------------------------------------------------------
# 1) SINGLE DILUTION (C1V1 = C2V2)
# ------------------------------------------------------------

def calc_single(
    stock_conc: float,
    target_conc: float,
    final_ul: float,
    vehicle_frac: float = 0.0,
) -> Dict[str, Any]:
    """
    Single dilution using C1 * V1 = C2 * V2.

    Parameters
    ----------
    stock_conc : float
        C1 (same units as target_conc, e.g. mM).
    target_conc : float
        C2 (same units as stock_conc).
    final_ul : float
        Final volume in µL.
    vehicle_frac : float
        Fraction of stock that is solvent (e.g. 1.0 for pure DMSO, 0.1 for 10% DMSO).

    Returns
    -------
    dict with add_stock_ul, add_solvent_ul, vehicle_percent
    """
    if stock_conc <= 0:
        raise ValueError("stock_conc must be > 0")
    if final_ul <= 0:
        raise ValueError("final_ul must be > 0")

    v1_ul = (target_conc * final_ul) / stock_conc
    solvent_ul = max(final_ul - v1_ul, 0.0)
    vehicle_percent = (v1_ul * vehicle_frac / final_ul) * 100.0

    return {
        "add_stock_ul": round(v1_ul, 4),
        "add_solvent_ul": round(solvent_ul, 4),
        "vehicle_percent": round(vehicle_percent, 6),
        "final_volume_ul": final_ul,
        "stock_conc": stock_conc,
        "target_conc": target_conc,
    }


# ------------------------------------------------------------
# 2) SERIAL DILUTION
# ------------------------------------------------------------

def calc_serial_dilution(
    start_conc: float,
    dilution_factor: float,
    steps: int,
    final_volume_ul: float,
    vehicle_frac: float = 0.0,
) -> Dict[str, Any]:
    """
    Create a serial dilution series.

    Returns a list of rows with:
    step, from_conc, to_conc, take_from_prev_ul, add_solvent_ul, vehicle_percent
    """
    if start_conc <= 0:
        raise ValueError("start_conc must be > 0")
    if dilution_factor <= 1.0:
        raise ValueError("dilution_factor must be > 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if final_volume_ul <= 0:
        raise ValueError("final_volume_ul must be > 0")

    rows = []
    current_conc = start_conc
    min_pip = 1.0  # µL

    for i in range(steps):
        next_conc = current_conc / dilution_factor
        v1_ul = (next_conc * final_volume_ul) / current_conc
        solvent_ul = final_volume_ul - v1_ul
        vehicle_percent = (v1_ul * vehicle_frac / final_volume_ul) * 100.0

        note = ""
        if v1_ul < min_pip:
            note = "<1 µL – consider intermediate stock"

        rows.append(
            {
                "step": i + 1,
                "from_conc": round(current_conc, 6),
                "to_conc": round(next_conc, 6),
                "take_from_prev_ul": round(v1_ul, 3),
                "add_solvent_ul": round(solvent_ul, 3),
                "vehicle_percent": round(vehicle_percent, 5),
                "note": note,
            }
        )
        current_conc = next_conc

    return {
        "start_conc": start_conc,
        "dilution_factor": dilution_factor,
        "steps": steps,
        "final_volume_ul": final_volume_ul,
        "rows": rows,
    }


# ------------------------------------------------------------
# 3) EXPERIMENT SERIES (PLATE-LIKE)
# ------------------------------------------------------------

def calc_experiment_series(
    final_concs_uM: List[float],
    stock_conc_uM: float,
    well_volume_ul: float,
    replicates: int = 1,
    overfill: float = 1.0,
    vehicle_frac: float = 1.0,
    max_vehicle_percent: float = 1.0,
) -> Dict[str, Any]:
    """
    Plate-like fixed-volume experiment series.

    Mirrors your UI logic: for each final conc:
      v_stock = C_final * V / C_stock
      v_medium = V - v_stock
      vehicle % = v_stock * vehicle_frac / V * 100
      total_vol_to_prepare = (v_stock + v_medium) * replicates * overfill
    """
    if stock_conc_uM <= 0:
        raise ValueError("stock_conc_uM must be > 0")
    if well_volume_ul <= 0:
        raise ValueError("well_volume_ul must be > 0")

    table = []
    for c in final_concs_uM:
        v1_ul = (c * well_volume_ul) / stock_conc_uM
        solvent_ul = well_volume_ul - v1_ul
        vehicle_percent = (v1_ul * vehicle_frac / well_volume_ul) * 100.0
        total_vol_ul = (v1_ul + solvent_ul) * replicates * overfill
        table.append(
            {
                "final_conc_uM": c,
                "add_stock_ul_per_well": round(v1_ul, 3),
                "add_medium_ul_per_well": round(solvent_ul, 3),
                "vehicle_percent": round(vehicle_percent, 5),
                "ok": vehicle_percent <= max_vehicle_percent,
                "total_volume_to_prepare_ul": round(total_vol_ul, 1),
            }
        )

    return {
        "stock_conc_uM": stock_conc_uM,
        "well_volume_ul": well_volume_ul,
        "replicates": replicates,
        "overfill": overfill,
        "rows": table,
    }


# ------------------------------------------------------------
# 4) FROM SOLID (MG → SOLUTION)
# ------------------------------------------------------------

def calc_solid_to_solution(
    mass_mg: float,
    mw_g_per_mol: float,
    target_conc: float,
    target_unit: str,
    final_volume_ml: float,
) -> Dict[str, Any]:
    """
    Calculate how much mass is needed to reach a given target concentration,
    and what stock molarity you get if you dissolve all available mass.

    target_unit: "uM" or "mM"
    """
    if mw_g_per_mol <= 0:
        raise ValueError("mw must be > 0")
    if final_volume_ml <= 0:
        raise ValueError("final_volume_ml must be > 0")

    if target_unit.lower() in ["µm", "um", "uM"]:
        C = target_conc * 1e-6  # mol/L
    else:
        C = target_conc * 1e-3  # mol/L

    V_L = final_volume_ml / 1000.0
    m_needed_g = C * V_L * mw_g_per_mol
    m_needed_mg = m_needed_g * 1000.0

    # If we dissolve all available mass in 1 mL / 2 mL
    mass_g_available = mass_mg / 1000.0
    n_mol = mass_g_available / mw_g_per_mol
    stock_if_1ml_mM = (n_mol / 0.001) * 1000.0
    stock_if_2ml_mM = (n_mol / 0.002) * 1000.0

    return {
        "mass_needed_mg": round(m_needed_mg, 3),
        "target_conc": target_conc,
        "target_unit": target_unit,
        "final_volume_ml": final_volume_ml,
        "if_all_dissolved_mM_in_1ml": round(stock_if_1ml_mM, 2),
        "if_all_dissolved_mM_in_2ml": round(stock_if_2ml_mM, 2),
    }


# ------------------------------------------------------------
# 5) UNIT CONVERTER (GENERAL) + mg/mL <-> mM
# ------------------------------------------------------------

def calc_unit_mgml_to_mM(mg_per_ml: float, mw: float) -> Dict[str, Any]:
    if mw <= 0:
        raise ValueError("mw must be > 0")
    return {"mM": (mg_per_ml * 1000.0) / mw}


def calc_unit_mM_to_mgml(mM: float, mw: float) -> Dict[str, Any]:
    if mw <= 0:
        raise ValueError("mw must be > 0")
    return {"mg_per_ml": (mM * mw) / 1000.0}


def calc_unit_converter(
    value: float,
    from_unit: str,
    to_unit: str,
    mw: float | None = None,
) -> Dict[str, Any]:
    """
    Simple generic unit converter. Currently supports:
    - mg/mL <-> mM (requires mw)
    - % w/v <-> g/L
    """
    fu = from_unit.lower()
    tu = to_unit.lower()

    if fu == "mg/ml" and tu == "mm":
        if mw is None:
            raise ValueError("mw is required for mg/mL → mM")
        return calc_unit_mgml_to_mM(value, mw)
    elif fu == "mm" and tu == "mg/ml":
        if mw is None:
            raise ValueError("mw is required for mM → mg/mL")
        return calc_unit_mM_to_mgml(value, mw)
    elif fu == "%" and tu in ["g/l", "g per l"]:
        g_per_l = value * 10.0  # % w/v: g per 100 mL → g/L
        return {"g_per_l": g_per_l}
    elif fu in ["g/l", "g per l"] and tu == "%":
        percent = value / 10.0
        return {"percent": percent}
    else:
        return {
            "input_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "mw": mw,
            "note": "Conversion not implemented for this unit combination.",
        }


# ------------------------------------------------------------
# 6) % SOLUTIONS
# ------------------------------------------------------------

def calc_percent_solution(
    percent: float,
    percent_type: str,
    final_volume_ml: float,
    stock_percent: float | None = None,
) -> Dict[str, Any]:
    """
    percent_type: 'w/v' or 'v/v'
    If stock_percent is given, we treat it as dilution from a % stock.
    """
    if final_volume_ml <= 0:
        raise ValueError("final_volume_ml must be > 0")

    if stock_percent is None:
        # prepare from solid / pure liquid
        if percent_type.lower().startswith("w"):
            grams_needed = (percent / 100.0) * final_volume_ml
            return {
                "grams_needed": round(grams_needed, 3),
                "ml_to_volume": final_volume_ml,
                "percent": percent,
                "type": "w/v",
            }
        else:
            ml_needed = (percent / 100.0) * final_volume_ml
            return {
                "solute_ml": round(ml_needed, 3),
                "final_volume_ml": final_volume_ml,
                "percent": percent,
                "type": "v/v",
            }
    else:
        # dilution from a % stock using C1V1 = C2V2
        V1 = (percent * final_volume_ml) / stock_percent
        return {
            "stock_percent": stock_percent,
            "target_percent": percent,
            "stock_volume_ml": round(V1, 3),
            "diluent_volume_ml": round(final_volume_ml - V1, 3),
            "percent_type": percent_type,
        }


# ------------------------------------------------------------
# 7) MOLARITY FROM MASS & VOLUME
# ------------------------------------------------------------

def calc_molarity_from_mass(
    mass_mg: float,
    mw_g_per_mol: float,
    volume_ml: float,
) -> Dict[str, Any]:
    if mw_g_per_mol <= 0:
        raise ValueError("mw must be > 0")
    if volume_ml <= 0:
        raise ValueError("volume_ml must be > 0")

    mass_g = mass_mg / 1000.0
    vol_L = volume_ml / 1000.0
    moles = mass_g / mw_g_per_mol
    molarity_M = moles / vol_L

    return {
        "M": molarity_M,
        "mM": molarity_M * 1000.0,
    }


# ------------------------------------------------------------
# 8) OD / CULTURE DILUTION
# ------------------------------------------------------------

def calc_od_culture_dilution(
    initial_od: float,
    target_od: float,
    final_volume_ml: float,
) -> Dict[str, Any]:
    if initial_od <= 0:
        raise ValueError("initial_od must be > 0")
    if target_od <= 0:
        raise ValueError("target_od must be > 0")
    if final_volume_ml <= 0:
        raise ValueError("final_volume_ml must be > 0")

    V1_ml = (target_od * final_volume_ml) / initial_od
    diluent_ml = final_volume_ml - V1_ml

    return {
        "culture_volume_ml": round(V1_ml, 3),
        "diluent_volume_ml": round(diluent_ml, 3),
        "final_volume_ml": final_volume_ml,
        "initial_od": initial_od,
        "target_od": target_od,
    }


# ------------------------------------------------------------
# 9) MASTER MIX / qPCR MIX
# ------------------------------------------------------------

def calc_master_mix(
    n_reactions: int,
    reaction_volume_ul: float,
    overfill: float,
    components_ul_per_rxn: Dict[str, float],
) -> Dict[str, Any]:
    """
    components_ul_per_rxn: e.g. {
        "buffer": 10.0,
        "dNTP": 0.0,
        "primer_f": 0.5,
        "primer_r": 0.5,
        "template": 1.0,
        "polymerase": 0.2,
    }
    """
    if n_reactions <= 0:
        raise ValueError("n_reactions must be > 0")
    if reaction_volume_ul <= 0:
        raise ValueError("reaction_volume_ul must be > 0")
    if overfill < 1.0:
        raise ValueError("overfill must be >= 1.0")

    per_rxn_sum = sum(components_ul_per_rxn.values())
    if per_rxn_sum > reaction_volume_ul:
        raise ValueError("Sum of components exceeds reaction volume.")

    total_rxn = n_reactions * overfill
    totals = {}
    for name, vol in components_ul_per_rxn.items():
        totals[name] = vol * total_rxn

    totals["water"] = max((reaction_volume_ul - per_rxn_sum) * total_rxn, 0.0)

    return {
        "n_reactions": n_reactions,
        "reaction_volume_ul": reaction_volume_ul,
        "overfill": overfill,
        "per_reaction_components": components_ul_per_rxn,
        "total_volumes_ul": totals,
    }


# ------------------------------------------------------------
# 10) MAKE X× STOCK FROM CURRENT STOCK
# ------------------------------------------------------------

def calc_make_fold_stock(
    current_conc: float,
    desired_multiple: float,
    final_volume_ml: float,
) -> Dict[str, Any]:
    if desired_multiple <= 1.0:
        raise ValueError("desired_multiple must be > 1")
    if final_volume_ml <= 0:
        raise ValueError("final_volume_ml must be > 0")

    V1_ml = final_volume_ml / desired_multiple
    solvent_ml = final_volume_ml - V1_ml

    return {
        "take_current_solution_ml": round(V1_ml, 3),
        "add_solvent_ml": round(solvent_ml, 3),
        "final_volume_ml": final_volume_ml,
        "fold": desired_multiple,
    }


# ------------------------------------------------------------
# 11) ACID / BASE DILUTION (COMMON REAGENTS)
# ------------------------------------------------------------

_COMMON_ACIDS = {
    "HCl_37": {"density": 1.19, "purity": 0.37, "mw": 36.46},
    "H2SO4_98": {"density": 1.84, "purity": 0.98, "mw": 98.08},
    "NH3_25": {"density": 0.91, "purity": 0.25, "mw": 17.03},
}

def calc_acid_base_dilution(
    reagent_key: str,
    target_molarity: float,
    final_volume_L: float,
) -> Dict[str, Any]:
    """
    reagent_key: one of 'HCl_37', 'H2SO4_98', 'NH3_25'
    """
    if reagent_key not in _COMMON_ACIDS:
        raise ValueError(f"Unknown reagent_key: {reagent_key}")
    if target_molarity <= 0:
        raise ValueError("target_molarity must be > 0")
    if final_volume_L <= 0:
        raise ValueError("final_volume_L must be > 0")

    r = _COMMON_ACIDS[reagent_key]
    moles_needed = target_molarity * final_volume_L
    mass_pure = moles_needed * r["mw"]
    mass_conc = mass_pure / r["purity"]
    vol_conc_L = mass_conc / r["density"]
    vol_conc_ml = vol_conc_L * 1000.0

    return {
        "reagent_key": reagent_key,
        "target_molarity": target_molarity,
        "final_volume_L": final_volume_L,
        "concentrated_volume_ml": round(vol_conc_ml, 2),
    }


# ------------------------------------------------------------
# 12) BUFFER HELPER (STATIC RECIPES)
# ------------------------------------------------------------

def calc_buffer_helper(buffer_type: str) -> Dict[str, Any]:
    recipes = {
        "PBS_1X_1L": {
            "NaCl_g": 8.0,
            "KCl_g": 0.2,
            "Na2HPO4_g": 1.44,
            "KH2PO4_g": 0.24,
            "volume_L": 1.0,
            "notes": "Dissolve in ~800 mL, adjust pH to 7.4, bring to 1 L.",
        },
        "PBS_10X_1L": {
            "NaCl_g": 80.0,
            "KCl_g": 2.0,
            "Na2HPO4_g": 14.4,
            "KH2PO4_g": 2.4,
            "volume_L": 1.0,
            "notes": "Dissolve, adjust pH, bring to 1 L.",
        },
        "TBS_1X_1L": {
            "NaCl_g": 8.0,
            "Tris_base_g": 3.0,
            "volume_L": 1.0,
            "notes": "Adjust pH 7.4–7.6 with HCl, bring to 1 L.",
        },
        "Tris_1M_pH8_1L": {
            "Tris_base_g": 121.14,
            "volume_L": 1.0,
            "notes": "Dissolve ~800 mL, adjust pH to 8.0 with HCl, bring to 1 L.",
        },
    }
    return recipes.get(buffer_type, {"note": "Unknown buffer type"})


# ------------------------------------------------------------
# 13) BEER–LAMBERT
# ------------------------------------------------------------

def calc_beer_lambert(
    absorbance: float | None = None,
    epsilon: float | None = None,
    concentration_M: float | None = None,
    path_length_cm: float | None = 1.0,
) -> Dict[str, Any]:
    """
    A = epsilon * c * l.
    Provide two of the three (A, epsilon, c) to solve for the third.
    """
    if path_length_cm is None or path_length_cm <= 0:
        raise ValueError("path_length_cm must be > 0")

    result: Dict[str, Any] = {
        "absorbance": absorbance,
        "epsilon": epsilon,
        "concentration_M": concentration_M,
        "path_length_cm": path_length_cm,
    }

    # solve for missing variable when possible
    if absorbance is not None and epsilon is not None and concentration_M is None:
        result["concentration_M"] = absorbance / (epsilon * path_length_cm)
    elif absorbance is not None and concentration_M is not None and epsilon is None:
        result["epsilon"] = absorbance / (concentration_M * path_length_cm)
    elif epsilon is not None and concentration_M is not None and absorbance is None:
        result["absorbance"] = epsilon * concentration_M * path_length_cm

    return result


# ------------------------------------------------------------
# 14) CELL SEEDING
# ------------------------------------------------------------

def calc_cell_seeding(
    cell_concentration_per_ml: float,
    target_cells_per_well: float,
    wells: int,
    final_volume_ml: float | None = None,
) -> Dict[str, Any]:
    if cell_concentration_per_ml <= 0:
        raise ValueError("cell_concentration_per_ml must be > 0")
    if target_cells_per_well <= 0:
        raise ValueError("target_cells_per_well must be > 0")
    if wells <= 0:
        raise ValueError("wells must be > 0")

    vol_cells_ml_per_well = target_cells_per_well / cell_concentration_per_ml

    return {
        "volume_cells_ml_per_well": vol_cells_ml_per_well,
        "wells": wells,
        "total_cells": target_cells_per_well * wells,
        "total_cells_volume_ml": vol_cells_ml_per_well * wells,
        "final_volume_ml_per_well": final_volume_ml,
    }


# ------------------------------------------------------------
# 15) DMSO PLATE CAP CHECKER
# ------------------------------------------------------------

def calc_dmso_plate_cap(
    final_concs_uM: List[float],
    stock_conc_uM: float,
    well_volume_ul: float,
    dmso_cap_percent: float,
    vehicle_frac: float = 1.0,
) -> Dict[str, Any]:
    rows = []
    for c in final_concs_uM:
        v1_ul = (c * well_volume_ul) / stock_conc_uM
        dmso_percent = (v1_ul * vehicle_frac / well_volume_ul) * 100.0
        rows.append(
            {
                "final_conc_uM": c,
                "stock_volume_ul": round(v1_ul, 3),
                "vehicle_percent": round(dmso_percent, 5),
                "ok": dmso_percent <= dmso_cap_percent,
            }
        )
    return {"rows": rows, "dmso_cap_percent": dmso_cap_percent}


# ------------------------------------------------------------
# 16) ALIQUOT SPLITTER
# ------------------------------------------------------------

def calc_aliquot_splitter(
    total_volume_ml: float,
    aliquot_volume_ml: float,
    dead_volume_ml: float = 0.0,
) -> Dict[str, Any]:
    usable_vol_ml = total_volume_ml - dead_volume_ml
    if usable_vol_ml <= 0:
        raise ValueError("Dead volume is >= total volume.")

    n_aliquots = math.floor(usable_vol_ml / aliquot_volume_ml)
    leftover = usable_vol_ml - n_aliquots * aliquot_volume_ml

    return {
        "total_volume_ml": total_volume_ml,
        "aliquot_volume_ml": aliquot_volume_ml,
        "dead_volume_ml": dead_volume_ml,
        "n_aliquots": n_aliquots,
        "leftover_volume_ml": round(leftover, 3),
    }


# ------------------------------------------------------------
# 17) PROTEIN EXTINCTION / MW FROM SEQUENCE
# ------------------------------------------------------------

_AA_MW = {
    "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}


def calc_protein_properties(sequence: str) -> Dict[str, Any]:
    seq = sequence.replace("\n", "").replace(" ", "").upper()
    length = len(seq)
    if length == 0:
        raise ValueError("Empty sequence")

    nW = seq.count("W")
    nY = seq.count("Y")
    nC = seq.count("C")
    nCystine = nC // 2  # crude

    epsilon = 5500 * nW + 1490 * nY + 125 * nCystine
    mw = sum(_AA_MW.get(aa, 110.0) for aa in seq) - (length - 1) * 18.0  # remove water for peptide bonds

    return {
        "length": length,
        "nW": nW,
        "nY": nY,
        "nCystine": nCystine,
        "epsilon": epsilon,
        "mw": mw,
    }


# ------------------------------------------------------------
# 18) pH & BUFFER CAPACITY (H-H)
# ------------------------------------------------------------

def calc_ph_buffer_capacity(
    pKa: float,
    acid_conc_mM: float,
    base_conc_mM: float,
) -> Dict[str, Any]:
    acid = acid_conc_mM / 1000.0
    base = base_conc_mM / 1000.0
    if acid <= 0 or base <= 0:
        raise ValueError("acid and base concentrations must be > 0")

    Ka = 10 ** (-pKa)
    ratio = base / acid
    pH = pKa + math.log10(ratio)

    H = 10 ** (-pH)
    Ctot = (acid + base)
    beta = 2.303 * Ctot * (Ka * H) / ((H + Ka) ** 2)

    return {
        "pH": pH,
        "buffer_capacity": beta,
        "pKa": pKa,
        "acid_mM": acid_conc_mM,
        "base_mM": base_conc_mM,
    }


# ------------------------------------------------------------
# 19) CELL CULTURE MEDIA DESIGNER (HEURISTIC)
# ------------------------------------------------------------

def calc_media_designer(
    base_media: str,
    cell_type: str | None = None,
    serum_percent: float | None = None,
    antibiotics: bool | None = None,
    glutamine: bool | None = None,
    supplements: List[str] | None = None,
) -> Dict[str, Any]:
    recipe: Dict[str, Any] = {"base_media": base_media}
    if supplements:
        recipe["supplements"] = supplements

    notes = []
    if cell_type == "HEK293":
        recipe.setdefault("serum_percent", serum_percent or 10)
        notes.append("Optional: 1 mM sodium pyruvate.")
    elif cell_type == "CHO":
        recipe.setdefault("serum_percent", serum_percent or 10)
    elif cell_type == "Drosophila S2":
        recipe.setdefault("serum_percent", max(serum_percent or 10, 10))
        notes.append("Use Schneider’s Drosophila Medium.")
    elif cell_type == "Primary neurons":
        recipe["base_media"] = "Neurobasal + B27"
        recipe["serum_percent"] = 0
        glutamine = True

    if antibiotics:
        recipe["antibiotics"] = "Pen/Strep 1×"
    if glutamine:
        recipe["glutamine"] = "2 mM"

    if notes:
        recipe["notes"] = notes

    return recipe


# ------------------------------------------------------------
# 20) PRIMER / PROBE CONCENTRATION HELPER
# ------------------------------------------------------------

def calc_primer_probe(
    stock_conc_uM: float,
    target_conc_uM: float,
    final_volume_ul: float,
) -> Dict[str, Any]:
    if stock_conc_uM <= 0:
        raise ValueError("stock_conc_uM must be > 0")
    if final_volume_ul <= 0:
        raise ValueError("final_volume_ul must be > 0")

    v_stock = (target_conc_uM * final_volume_ul) / stock_conc_uM
    v_buffer = final_volume_ul - v_stock

    return {
        "stock_volume_ul": round(v_stock, 3),
        "buffer_volume_ul": round(v_buffer, 3),
        "target_conc_uM": target_conc_uM,
        "final_volume_ul": final_volume_ul,
    }


# ------------------------------------------------------------
# 21) INVENTORY TRACKER (STATELESS STUB)
# ------------------------------------------------------------

def calc_inventory_tracker(
    action: str,
    reagent_name: str | None = None,
    amount: float | None = None,
    unit: str | None = None,
) -> Dict[str, Any]:
    """
    Stateless stub. Real inventory should use DB; here we just echo the request.
    """
    return {
        "action": action,
        "reagent_name": reagent_name,
        "amount": amount,
        "unit": unit,
        "note": "Inventory operations should be implemented against a database.",
    }


# ------------------------------------------------------------
# 22) REAGENT STABILITY PREDICTOR
# ------------------------------------------------------------

def calc_reagent_stability_predictor(
    reagent_name: str,
    temperature_c: float,
    light_exposed: bool | None = None,
) -> Dict[str, Any]:
    t = reagent_name.lower()
    tips: List[str] = []

    if "retinal" in t or "retinoic" in t:
        tips.append("Light-sensitive: wrap in foil/amber, use dry solvent, store at -20°C.")
    if "dye" in t or "fluorescein" in t or "rhodamine" in t:
        tips.append("Avoid repeated freeze–thaw; aliquot and protect from light.")
    if "enzyme" in t:
        tips.append("Keep on ice during setup; consider glycerol for stability.")
    if light_exposed:
        tips.append("Minimize light exposure where possible.")

    if not tips:
        tips.append("No specific rule matched. Use general best practices (4°C short term, -20°C long term).")

    return {
        "reagent_name": reagent_name,
        "temperature_c": temperature_c,
        "light_exposed": light_exposed,
        "tips": tips,
    }


# ------------------------------------------------------------
# 23) DILUTION SERIES VISUALIZER (DATA ONLY)
# ------------------------------------------------------------

def calc_dilution_series_visualizer(
    start_conc: float,
    dilution_factor: float,
    steps: int,
) -> Dict[str, Any]:
    if start_conc <= 0 or dilution_factor <= 1.0 or steps <= 0:
        raise ValueError("Invalid parameters for dilution series")

    concs = [start_conc / (dilution_factor ** i) for i in range(steps)]
    rows = [{"step": i + 1, "concentration": concs[i]} for i in range(steps)]
    return {"rows": rows}


# ------------------------------------------------------------
# 24) LAB NOTEBOOK GENERATOR (STRUCTURED CONTENT)
# ------------------------------------------------------------

def calc_lab_notebook_generator(
    title: str,
    date: str,
    summary: str,
    steps: List[str],
) -> Dict[str, Any]:
    content_md = f"# {title}\n\n"
    content_md += f"**Date:** {date}\n\n"
    content_md += "## Summary\n" + summary + "\n\n## Steps\n"
    for s in steps:
        content_md += f"- {s}\n"

    return {
        "title": title,
        "date": date,
        "summary": summary,
        "steps": steps,
        "markdown": content_md,
    }


# ------------------------------------------------------------
# 25) OSMOLARITY CALCULATOR (SINGLE SOLUTE)
# ------------------------------------------------------------

def calc_osmolarity(
    molarity_M: float,
    dissociation_particles: int,
) -> Dict[str, Any]:
    """
    Approximate osmolarity for a single solute:
      Osm (Osm/L) = molarity * i
    """
    if molarity_M < 0 or dissociation_particles <= 0:
        raise ValueError("Invalid parameters for osmolarity")

    osm = molarity_M * dissociation_particles
    return {
        "osmolarity_Osm_per_L": osm,
        "osmolarity_mOsm_per_L": osm * 1000.0,
    }


# ------------------------------------------------------------
# 26) SPECTROPHOTOMETRY TOOLBOX (SIMPLE OPS)
# ------------------------------------------------------------

def calc_spectrophotometry_toolbox(
    operation: str,
    values: List[float],
) -> Dict[str, Any]:
    if not values:
        raise ValueError("values cannot be empty")

    if operation == "A_to_T":
        # convert absorbance to percent transmittance: T% = 10^(-A) * 100
        res = [10 ** (-A) * 100.0 for A in values]
        return {"operation": operation, "input": values, "output": res}
    elif operation == "normalize":
        max_val = max(values)
        if max_val == 0:
            return {"operation": operation, "input": values, "output": values}
        res = [v / max_val for v in values]
        return {"operation": operation, "input": values, "output": res}
    elif operation == "ratio":
        if len(values) < 2:
            raise ValueError("ratio requires at least two values")
        ratios = [values[i] / values[0] if values[0] != 0 else None for i in range(len(values))]
        return {"operation": operation, "input": values, "output": ratios}
    else:
        return {"operation": operation, "input": values, "note": "Unknown operation"}


# ------------------------------------------------------------
# 27) SOLUTION DENSITY CONVERTER
# ------------------------------------------------------------

_DENSITY = {
    "water": 1.0,
    "ethanol_100%": 0.789,
    "ethanol_70%": 0.88,
    "glycerol_100%": 1.26,
    "dmso_100%": 1.10,
    "acetone_100%": 0.788,
}

def calc_solution_density_converter(
    density_g_per_ml: float,
    value: float,
    from_unit: str,
    to_unit: str,
    mw: float | None = None,
) -> Dict[str, Any]:
    """
    Simple helper:
    - "% w/v" <-> g/L and M (approx, using mw)
    """
    fu = from_unit.lower()
    tu = to_unit.lower()

    if fu == "% w/v" and tu in ["g/l", "g per l"]:
        g_per_l = value * 10.0
        return {"g_per_l": g_per_l}
    elif fu in ["g/l", "g per l"] and tu == "% w/v":
        percent = value / 10.0
        return {"percent": percent}
    elif fu in ["g/l", "g per l"] and tu == "m" and mw is not None:
        M = value / mw
        return {"M": M}
    elif fu == "m" and tu in ["g/l", "g per l"] and mw is not None:
        g_per_l = value * mw
        return {"g_per_l": g_per_l}
    else:
        return {
            "density_g_per_ml": density_g_per_ml,
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "mw": mw,
            "note": "Conversion not implemented.",
        }


# ------------------------------------------------------------
# 28) REAGENT COMPATIBILITY CHECKER (RULE-BASED)
# ------------------------------------------------------------

def calc_reagent_compatibility_checker(
    reagent_a: str,
    reagent_b: str,
) -> Dict[str, Any]:
    text_a = reagent_a.lower()
    text_b = reagent_b.lower()
    warnings: List[str] = []

    pair = f"{text_a} + {text_b}"

    if ("phosphate" in pair and "calcium" in pair) or ("pbs" in pair and "cacl2" in pair):
        warnings.append("Phosphate + calcium can precipitate (Ca3(PO4)2).")
    if "dmso" in pair and "aqueous" in pair:
        warnings.append("Avoid adding high % DMSO directly to aqueous media – pre-dilute.")
    if "ethanol" in pair and "protein" in pair:
        warnings.append("High % ethanol can denature proteins.")

    return {
        "reagent_a": reagent_a,
        "reagent_b": reagent_b,
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }


# ------------------------------------------------------------
# 29) STORAGE / STABILITY HELPER (SIMPLE MAP)
# ------------------------------------------------------------

def calc_storage_stability_helper(
    name: str,
) -> Dict[str, Any]:
    storage_rules = {
        "retinal": "Protect from light, dissolve in dry EtOH or DMSO, aliquot, store at -20°C or below.",
        "retinoic": "Light-sensitive, store at -20°C, use fresh aliquots.",
        "ampicillin": "Store stock at -20°C, avoid repeated freeze–thaw.",
        "pbs": "Store at RT or 4°C for ~1 month.",
        "tris": "Store at RT for ~1 month.",
        "pfa": "Store at 4°C, protected from light; discard if precipitate forms.",
    }
    lname = name.lower()
    for key, rule in storage_rules.items():
        if key in lname:
            return {"name": name, "advice": rule}
    return {
        "name": name,
        "advice": "No specific rule found. General rule: 4°C for short term, -20°C for long term, protect from light if colored/retinoid.",
    }


# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------

CALC_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    # your existing ones (used by batch + old chat)
    "single_dilution": calc_single,
    "mgml_to_mM": calc_unit_mgml_to_mM,
    "mM_to_mgml": calc_unit_mM_to_mgml,

    # new names used by llm_tools / Tier-5 LLM tools
    "serial_dilution": calc_serial_dilution,
    "experiment_series": calc_experiment_series,
    "solid_to_solution": calc_solid_to_solution,
    "unit_converter": calc_unit_converter,
    "percent_solution": calc_percent_solution,
    "molarity_from_mass": calc_molarity_from_mass,
    "od_culture_dilution": calc_od_culture_dilution,
    "master_mix": calc_master_mix,
    "make_fold_stock": calc_make_fold_stock,
    "acid_base_dilution": calc_acid_base_dilution,
    "buffer_helper": calc_buffer_helper,
    "beer_lambert": calc_beer_lambert,
    "cell_seeding": calc_cell_seeding,
    "dmso_plate_cap": calc_dmso_plate_cap,
    "aliquot_splitter": calc_aliquot_splitter,
    "protein_properties": calc_protein_properties,
    "ph_buffer_capacity": calc_ph_buffer_capacity,
    "media_designer": calc_media_designer,
    "primer_probe_calc": calc_primer_probe,
    "inventory_tracker": calc_inventory_tracker,
    "reagent_stability_predictor": calc_reagent_stability_predictor,
    "dilution_series_visualizer": calc_dilution_series_visualizer,
    "lab_notebook_generator": calc_lab_notebook_generator,
    "osmolarity_calc": calc_osmolarity,
    "spectrophotometry_toolbox": calc_spectrophotometry_toolbox,
    "solution_density_converter": calc_solution_density_converter,
    "reagent_compatibility_checker": calc_reagent_compatibility_checker,
    "storage_stability_helper": calc_storage_stability_helper,
}
