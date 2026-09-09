# Reproduce the Pertpy evaluator demonstration

The public audit branch includes executable tests and the data preparation recipe.
The clean contribution contains only five implementation, test and documentation files.

These commands target Linux. The successful evidence used Python 3.12.3 and
3.14.7. The real-data example ran on 3.12.3. Save the accompanying
`requirements-linux-3.12.txt` before running the following commands.

```sh
git clone https://github.com/thantiklermcirony/pertpy.git
cd pertpy
git checkout 1ce7cab302b30077cc283fabc3cebd5c4a18c92b
uv venv --python 3.12.3
uv pip install -r /path/to/requirements-linux-3.12.txt
uv pip install --no-deps --no-sources -e .
.venv/bin/python -m pytest tests/tools/test_perturbation_evaluator.py tests/tools/independent_evaluator_test.py -v
.venv/bin/python -m pytest tests/tools/_perturbation_space/test_comparison.py -v
.venv/bin/python tests/tools/download_figshare_norman.py
.venv/bin/python tests/tools/prepare_norman_subset.py tests/tools/Norman_2019.h5ad --output tests/tools/subset.h5ad
.venv/bin/python tests/tools/run_norman_evaluator.py tests/tools/subset.h5ad --output results
```

The download is 1,703,064,678 bytes. Preparation validates SHA-256
`b679c157fea550ef5be4dad91da9dff1f5d8287313ad3f3537f6cf14d4bcd434`,
checks guide labels and quality flags, selects conditions/cells by deterministic
identifier hashes, and fits gene selection on training cells only. It refuses
existing output files; use a fresh output path for a second preparation.

The source is the processed [Theis lab Norman dataset](https://github.com/theislab/sc-pert/blob/main/data_table.csv)
from [Norman et al., Science 2019](https://doi.org/10.1126/science.aax4438).
Per-cell normalization is log1p of counts normalized to 10,000 using the full
19,018-gene archived denominator. The archived global gene filtering is inherited.
The result is a software demonstration in one K562 context, not an independent
clinical or biological validation. Technical lanes are not biological replicates.

Only recipes and aggregate results are distributed here. The exact Figshare
object's redistribution license has not been established; no H5AD is included.
Floating-point values can vary by a few ulps across runs. Two Linux runs differed
by at most 2.22e-16 in their scores; comparison with a separate `math.fsum` oracle
also agreed within 2.22e-16. The internal independent MSE/E-distance checks use
relative/absolute tolerance 1e-10, and the external MSE/Pearson check uses 1e-12/1e-14.

For developer checks, the audit workflow assembles the exact five contribution
files over upstream commit `5bf4acaa45af16b49d248e4986fe394344fafdf5`, installs the repository's declared developer
environment, runs `mypy src/pertpy tests`, its applicable pre-commit hooks, and
`sphinx-build -M html docs docs/_build -W --keep-going`.
