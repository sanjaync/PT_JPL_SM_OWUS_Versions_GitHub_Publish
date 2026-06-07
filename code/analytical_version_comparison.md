# Comparison: Analytical vs. Empirical OWUS

The newer **Analytical OWUS** versions (`v12.3.1.1` and `v13.3.1.1`) represent a major leap from the older empirical versions in three specific ways: **Physical Realism**, **Parameter Meaning**, and **Numerical Robustness**.

## 1. From "Shapes" to "Physics"
*   **Old Version (Empirical)**: You had to choose a geometric "shape" for how plants respond to water stress (Linear, Exponential, or Sigmoidal). These were just mathematical curves that happened to fit the data but didn't know *why* the plant was stressed.
*   **New Version (Analytical)**: The curve is **mechanistically derived** from the Soil-Plant-Atmosphere Continuum (SPAC) equations. The "sigmoidal" look isn't a choice; it's the physical result of how xylem resistance and stomatal closure interact.

## 2. Parameters with Biological Meaning
*   **Old Version**: Used abstract fitting parameters (a, b, c) that had no units and couldn't be measured in the field.
*   **New Version**: Uses **Pi-groups** (pi_R, pi_F, pi_T, pi_S) that represent real plant traits:
    *   **pi_R**: Plant-to-soil resistance ratio.
    *   **pi_T**: Sensitivity to transpiration demand.
    *   This allows the model to be "trait-aware"—if you know a site has deep roots or high xylem vulnerability, you can set the parameters based on biology, not just curve-fitting.

## 3. Integrated "Dynamic" Plasticity
*   **Old Version**: Handling "Dynamic" changes (where plants adjust to drought over time) was often clunky and prone to indexing errors. 
*   **New Version**: Full vectorization for the dynamic solver. The code now identifies unique biological "regimes" across the time series and solves the analytical inversion in batches. This makes it significantly faster and more stable.

## 4. Automatic "Well-Watered" Plateau (f_ww)
*   In the old code, you often had to assume the plant could always reach 100% transpiration if soil was wet.
*   The new code calculates a physical **Transpiration Maximum (f_ww)**. Even in wet soil, if the plant's internal resistance is too high or the atmosphere is too dry, the model correctly predicts that the plant *cannot* reach full potential.

| Feature | Old Versions (v12.3.1 / v13.3.1) | New Analytical (v12.3.1.1 / v13.3.1.1) |
| :--- | :--- | :--- |
| **Foundation** | Curve-fitting (Empirical) | SPAC Physics (Mechanistic) |
| **Site Response** | Manually chosen (Linear/Exp/Sig) | Physically derived (Single Solver) |
| **Parameters** | Math coefficients (a, b, c) | Biological traits (pi-groups) |
| **Solver** | Simple grid search | Analytical inversion + Interpolation |
| **Robustness** | High failure risk on "unseen" sites | More stable across different climates |
