# CFBD Advanced Stats Pull Summary

## Pull Details
- **Years pulled:** 2014‑2025 (inclusive)
- **Number of API calls:** 12 (one per year)
- **Files committed:** `data/cfbd/advanced_<year>.json` for each year in the range.
- **Command used:** 
  ```bash
  python -m cfb_predictor.data.cfbd_advanced --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
  ```
  (after fixing the `cfb.Configuration` typo to `cfb.Config`).

## Block Evaluation Notes
The current feature‑building pipeline in `cfb_predictor/features/build.py` does **not** accept auxiliary inputs (efficiency, priors, etc.). Consequently:
- The **EPA block** cannot be evaluated because it requires efficiency auxiliaries that are not wired into the training/fold generation.
- The **priors block** is present in the code but is not connected to the training pipeline (it is “inert”); therefore it cannot be evaluated either.
- No block evaluation was performed for CFB; this PR documents the pulled JSON files and notes the lack of aux support as the reason.

## Files
The following JSON files are present in `data/cfbd/`:
- advanced_2014.json
- advanced_2015.json
- advanced_2016.json
- advanced_2017.json
- advanced_2018.json
- advanced_2019.json
- advanced_2020.json
- advanced_2021.json
- advanced_2022.json
- advanced_2023.json
- advanced_2024.json
- advanced_2025.json

Each file contains the CFBD advanced stats for the respective season.

## Verification
The pull respected the budget of 40 API calls, using exactly 12.
