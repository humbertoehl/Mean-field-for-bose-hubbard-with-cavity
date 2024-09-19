# Bose-Hubbard Model with Cavity Interaction by Mean-Field Approach

This repository performs calculations of a Bose-Hubbard model extended with a cavity interaction using a mean-field approach. The main script is `PhaseDiagram.py`.

## Usage

Run the script:
```bash
python PhaseDiagram.py
```

You'll be prompted to select one of the following operations:
1. **Mean-field parameters for a point**: Input interaction terms (`mu/U`, `zt/U`, `U_cav/U`) to find the optimal parameters.
2. **Mean-field parameters as functions of `mu`**: Input interaction terms and `mu` range to plot the parameters.
3. **Phase diagram**: Input ranges for `zt/U` and `mu/U` to generate a full phase diagram.
