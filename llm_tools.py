# llm_tools.py

LLM_TOOLS = [
    # 1. single_dilution
    {
        "type": "function",
        "function": {
            "name": "single_dilution",
            "description": "Use C1V1 = C2V2 to dilute a stock solution to a desired concentration and volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_conc": {"type": "number", "description": "Stock concentration (same units as target, e.g. mM)."},
                    "target_conc": {"type": "number", "description": "Final desired concentration (same units as stock)."},
                    "final_volume": {"type": "number", "description": "Final volume (mL or µL)."},
                    "volume_unit": {"type": "string", "enum": ["ul", "mL"], "description": "Unit of final_volume."}
                },
                "required": ["stock_conc", "target_conc", "final_volume", "volume_unit"]
            }
        }
    },

    # 2. serial_dilution
    {
        "type": "function",
        "function": {
            "name": "serial_dilution",
            "description": "Create a serial dilution series (e.g. 1:10 across N tubes/plate wells).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_conc": {"type": "number", "description": "Starting concentration (e.g. mg/mL or mM)."},
                    "dilution_factor": {"type": "number", "description": "Fold-dilution each step (e.g. 10 for 1:10)."},
                    "steps": {"type": "integer", "description": "Number of dilution steps."},
                    "final_volume": {"type": "number", "description": "Volume for each tube/well."},
                    "volume_unit": {"type": "string", "enum": ["ul", "mL"], "description": "Unit of final_volume."}
                },
                "required": ["start_conc", "dilution_factor", "steps", "final_volume", "volume_unit"]
            }
        }
    },

    # 3. experiment_series (plate-like)
    {
        "type": "function",
        "function": {
            "name": "experiment_series",
            "description": "Design a plate-like experiment series (dose-response, gradients, plate map).",
            "parameters": {
                "type": "object",
                "properties": {
                    "plate_type": {"type": "string", "enum": ["96", "48", "24", "12", "6"], "description": "Plate size in wells."},
                    "min_conc": {"type": "number", "description": "Minimum concentration."},
                    "max_conc": {"type": "number", "description": "Maximum concentration."},
                    "steps": {"type": "integer", "description": "Number of unique concentration steps."},
                    "replicates": {"type": "integer", "description": "Replicates per condition."}
                },
                "required": ["plate_type", "min_conc", "max_conc", "steps"]
            }
        }
    },

    # 4. solid_to_solution
    {
        "type": "function",
        "function": {
            "name": "solid_to_solution",
            "description": "Calculate mass of solid needed to make a target concentration solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_conc": {"type": "number", "description": "Target concentration (mg/mL or % w/v)."},
                    "conc_unit": {"type": "string", "enum": ["mg_per_ml", "percent_wv"], "description": "Unit for target_conc."},
                    "final_volume_ml": {"type": "number", "description": "Final volume in mL."}
                },
                "required": ["target_conc", "conc_unit", "final_volume_ml"]
            }
        }
    },

    # 5. unit_converter
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert between mg/mL, mM, %, g/L, ppm, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert."},
                    "from_unit": {"type": "string", "description": "Source unit, e.g. 'mg/ml', 'mM', '%', 'g/L', 'ppm'."},
                    "to_unit": {"type": "string", "description": "Target unit, e.g. 'mM', 'mg/ml', '%', 'µM'."},
                    "mw": {"type": "number", "description": "Molecular weight (if needed)."}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },

    # 6. percent_solution
    {
        "type": "function",
        "function": {
            "name": "percent_solution",
            "description": "Calculate how to make a % w/v or % v/v solution from solids or stock liquids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {"type": "number", "description": "Percent value, e.g. 2 for 2%."},
                    "percent_type": {"type": "string", "enum": ["w/v", "v/v"], "description": "Percent type."},
                    "final_volume_ml": {"type": "number", "description": "Final volume in mL."},
                    "stock_percent": {"type": "number", "description": "If diluting from a % stock, its percent value.", "nullable": True}
                },
                "required": ["percent", "percent_type", "final_volume_ml"]
            }
        }
    },

    # 7. molarity_from_mass
    {
        "type": "function",
        "function": {
            "name": "molarity_from_mass",
            "description": "Calculate molarity from mass, volume and molecular weight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mass_mg": {"type": "number", "description": "Mass in mg."},
                    "volume_ml": {"type": "number", "description": "Volume in mL."},
                    "mw": {"type": "number", "description": "Molecular weight in g/mol."}
                },
                "required": ["mass_mg", "volume_ml", "mw"]
            }
        }
    },

    # 8. od_culture_dilution
    {
        "type": "function",
        "function": {
            "name": "od_culture_dilution",
            "description": "Dilute a culture from one OD to another to a desired final volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "initial_od": {"type": "number", "description": "Starting OD (e.g. 1.2)."},
                    "target_od": {"type": "number", "description": "Target OD."},
                    "final_volume_ml": {"type": "number", "description": "Final volume in mL."}
                },
                "required": ["initial_od", "target_od", "final_volume_ml"]
            }
        }
    },

    # 9. master_mix
    {
        "type": "function",
        "function": {
            "name": "master_mix",
            "description": "Design PCR/qPCR master mix for N reactions with overage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction_volume_ul": {"type": "number", "description": "Volume per reaction in µL."},
                    "reactions": {"type": "integer", "description": "Number of reactions."},
                    "overage_fraction": {"type": "number", "description": "Extra fraction, e.g. 0.1 for 10%."}
                },
                "required": ["reaction_volume_ul", "reactions"]
            }
        }
    },

    # 10. make_fold_stock
    {
        "type": "function",
        "function": {
            "name": "make_fold_stock",
            "description": "Make an X-fold stock from a 1X recipe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fold": {"type": "number", "description": "Fold-concentration (e.g. 10 for 10X)."},
                    "final_volume_ml": {"type": "number", "description": "Final stock volume in mL."}
                },
                "required": ["fold", "final_volume_ml"]
            }
        }
    },

    # 11. acid_base_dilution
    {
        "type": "function",
        "function": {
            "name": "acid_base_dilution",
            "description": "Dilute concentrated acid or base to a target molarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_molarity": {"type": "number", "description": "Stock acid/base molarity."},
                    "target_molarity": {"type": "number", "description": "Target molarity."},
                    "final_volume_ml": {"type": "number", "description": "Final volume in mL."}
                },
                "required": ["stock_molarity", "target_molarity", "final_volume_ml"]
            }
        }
    },

    # 12. buffer_helper
    {
        "type": "function",
        "function": {
            "name": "buffer_helper",
            "description": "Help prepare common buffers (PBS, TBS, Tris, etc.) at a given molarity and pH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buffer_type": {"type": "string", "description": "e.g. 'PBS', 'TBS', 'Tris', 'HEPES'."},
                    "molarity": {"type": "number", "description": "Buffer molarity in M."},
                    "volume_l": {"type": "number", "description": "Volume in L."},
                    "target_pH": {"type": "number", "description": "Target pH, if relevant.", "nullable": True}
                },
                "required": ["buffer_type", "molarity", "volume_l"]
            }
        }
    },

    # 13. beer_lambert
    {
        "type": "function",
        "function": {
            "name": "beer_lambert",
            "description": "Apply Beer-Lambert law (A = ε·c·l) to compute concentration or absorbance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "absorbance": {"type": "number", "description": "Measured absorbance.", "nullable": True},
                    "epsilon": {"type": "number", "description": "Molar extinction coefficient (M-1 cm-1).", "nullable": True},
                    "concentration": {"type": "number", "description": "Concentration in M.", "nullable": True},
                    "path_length_cm": {"type": "number", "description": "Path length in cm.", "nullable": True}
                },
                "required": []
            }
        }
    },

    # 14. cell_seeding
    {
        "type": "function",
        "function": {
            "name": "cell_seeding",
            "description": "Calculate volumes and cell numbers for seeding plates/flasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_concentration_per_ml": {"type": "number", "description": "Current cell density (cells/mL)."},
                    "target_cells_per_well": {"type": "number", "description": "Desired cells per well/flask."},
                    "wells": {"type": "integer", "description": "Number of wells/flasks."}
                },
                "required": ["cell_concentration_per_ml", "target_cells_per_well", "wells"]
            }
        }
    },

    # 15. dmso_plate_cap
    {
        "type": "function",
        "function": {
            "name": "dmso_plate_cap",
            "description": "Check final DMSO fraction in wells given stock volume and well volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dmso_stock_fraction": {"type": "number", "description": "DMSO fraction in stock (e.g. 1 for pure DMSO)."},
                    "stock_volume_ul": {"type": "number", "description": "Volume of stock added per well (µL)."},
                    "well_volume_ul": {"type": "number", "description": "Final well volume (µL)."},
                    "max_dmso_fraction": {"type": "number", "description": "Max allowed DMSO fraction, e.g. 0.01.", "nullable": True}
                },
                "required": ["dmso_stock_fraction", "stock_volume_ul", "well_volume_ul"]
            }
        }
    },

    # 16. aliquot_splitter
    {
        "type": "function",
        "function": {
            "name": "aliquot_splitter",
            "description": "Split a sample (volume or mass) into aliquots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_volume_ul": {"type": "number", "description": "Total volume in µL.", "nullable": True},
                    "total_mass_mg": {"type": "number", "description": "Total mass in mg.", "nullable": True},
                    "aliquots": {"type": "integer", "description": "Number of aliquots."}
                },
                "required": ["aliquots"]
            }
        }
    },

    # 17. storage_stability
    {
        "type": "function",
        "function": {
            "name": "storage_stability",
            "description": "Give storage/stability guidance for a reagent or solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reagent_name": {"type": "string", "description": "Name of reagent or solution."},
                    "temperature_c": {"type": "number", "description": "Storage temperature (°C)."}
                },
                "required": ["reagent_name"]
            }
        }
    },

    # 18. protein_properties
    {
        "type": "function",
        "function": {
            "name": "protein_properties",
            "description": "Calculate protein MW, extinction coefficient, etc. from sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "string", "description": "Protein amino acid sequence (one-letter)."}
                },
                "required": ["sequence"]
            }
        }
    },

    # 19. ph_buffer_capacity
    {
        "type": "function",
        "function": {
            "name": "ph_buffer_capacity",
            "description": "Calculate effect of adding acid/base on buffer pH (Henderson–Hasselbalch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pKa": {"type": "number", "description": "pKa of buffer system."},
                    "acid_conc": {"type": "number", "description": "Concentration of HA."},
                    "base_conc": {"type": "number", "description": "Concentration of A-."}
                },
                "required": ["pKa", "acid_conc", "base_conc"]
            }
        }
    },

    # 20. media_designer
    {
        "type": "function",
        "function": {
            "name": "media_designer",
            "description": "Design a culture medium with supplements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_media": {"type": "string", "description": "e.g. DMEM, RPMI, LB, etc."},
                    "supplements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of supplements, e.g. '10% FBS', '2 mM glutamine'."
                    }
                },
                "required": ["base_media"]
            }
        }
    },

    # 21. primer_probe_calc
    {
        "type": "function",
        "function": {
            "name": "primer_probe_calc",
            "description": "Calculate primer/probe dilutions for PCR/qPCR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_conc_uM": {"type": "number", "description": "Primer/probe stock concentration in µM."},
                    "target_conc_uM": {"type": "number", "description": "Target working concentration in µM."},
                    "final_volume_ul": {"type": "number", "description": "Final volume in µL."}
                },
                "required": ["stock_conc_uM", "target_conc_uM", "final_volume_ul"]
            }
        }
    },

    # 22. inventory_tracker
    {
        "type": "function",
        "function": {
            "name": "inventory_tracker",
            "description": "Perform inventory operations: add, update, list reagents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "update", "list", "consume"], "description": "Inventory action."},
                    "reagent_name": {"type": "string", "description": "Reagent name.", "nullable": True},
                    "amount": {"type": "number", "description": "Amount to add/update/consume.", "nullable": True},
                    "unit": {"type": "string", "description": "Unit for amount.", "nullable": True}
                },
                "required": ["action"]
            }
        }
    },

    # 23. reagent_stability_predictor
    {
        "type": "function",
        "function": {
            "name": "reagent_stability_predictor",
            "description": "Heuristic stability prediction for a reagent under given conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reagent_name": {"type": "string"},
                    "temperature_c": {"type": "number"},
                    "light_exposed": {"type": "boolean", "nullable": True}
                },
                "required": ["reagent_name", "temperature_c"]
            }
        }
    },

    # 24. dilution_series_visualizer
    {
        "type": "function",
        "function": {
            "name": "dilution_series_visualizer",
            "description": "Generate data for plotting a dilution series.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_conc": {"type": "number"},
                    "dilution_factor": {"type": "number"},
                    "steps": {"type": "integer"}
                },
                "required": ["start_conc", "dilution_factor", "steps"]
            }
        }
    },

    # 25. lab_notebook_generator
    {
        "type": "function",
        "function": {
            "name": "lab_notebook_generator",
            "description": "Generate a formatted notebook entry (markdown/PDF-ready) for an experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string", "description": "Date string, e.g. '2025-11-17'."},
                    "summary": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "summary"]
            }
        }
    },

    # 26. osmolarity_calc
    {
        "type": "function",
        "function": {
            "name": "osmolarity_calc",
            "description": "Calculate osmolarity of a solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "molarity": {"type": "number", "description": "Molarity in M."},
                    "dissociation_particles": {"type": "integer", "description": "Number of particles per formula unit (e.g. 2 for NaCl)."}
                },
                "required": ["molarity", "dissociation_particles"]
            }
        }
    },

    # 27. spectrophotometry_toolbox
    {
        "type": "function",
        "function": {
            "name": "spectrophotometry_toolbox",
            "description": "Simple spectrophotometry utilities (e.g., A→T%, peak ratios).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "e.g. 'A_to_T', 'ratio', 'normalize'."},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "List of numeric values."}
                },
                "required": ["operation", "values"]
            }
        }
    },

    # 28. solution_density_converter
    {
        "type": "function",
        "function": {
            "name": "solution_density_converter",
            "description": "Convert between w/w, w/v, molarity using density.",
            "parameters": {
                "type": "object",
                "properties": {
                    "density_g_per_ml": {"type": "number"},
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                    "mw": {"type": "number", "nullable": True}
                },
                "required": ["density_g_per_ml", "value", "from_unit", "to_unit"]
            }
        }
    },

    # 29. reagent_compatibility_checker
    {
        "type": "function",
        "function": {
            "name": "reagent_compatibility_checker",
            "description": "Check whether two reagents/conditions are likely compatible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reagent_a": {"type": "string"},
                    "reagent_b": {"type": "string"}
                },
                "required": ["reagent_a", "reagent_b"]
            }
        }
    }
]
