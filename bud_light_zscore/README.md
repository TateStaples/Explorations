# The Bud Light Demand Shock: a z-score analysis

Quantifies how far Bud Light / Anheuser-Busch InBev U.S. revenue fell after the
1 April 2023 Dylan Mulvaney controversy, in units of its own pre-controversy
standard deviation.

## Headline

Against a six-quarter pre-controversy baseline of **+2.63% ± 0.74 pp** YoY
organic revenue growth:

| Quarter | Growth | z | p (small-sample corrected) |
|---|---|---|---|
| 2Q23 | −10.5% | **−17.8** | < 1e−4 |
| 3Q23 | −13.5% | **−21.9** | < 1e−4 |
| 4Q23 | −17.3% | **−27.1** | < 1e−4 |
| 1Q24 | −9.1%  | **−15.9** | < 1e−4 |
| 2Q24 | −0.6%  | −4.4 | 0.0097 |
| 4Q24 | +0.8%  | −2.5 | 0.0694 |

Permutation test on the pre/post split: p = 0.0010. Brand-level NIQ scanner
data puts Bud Light off-premise dollar sales on a −25.5% plateau within six
weeks, implying |z| > 6 under any plausible baseline assumption.

The bootstrap 95% interval on the trough z is roughly [−97, −20] — the
direction and persistence are solid, the second digit of z is not. The report
says so explicitly.

## Important caveat

**AB InBev does not disclose brand-level revenue for Bud Light.** The primary
series is AB InBev's reported U.S. organic revenue growth (the segment Bud Light
dominates, and which the company itself attributes to Bud Light volume declines);
NIQ weekly scanner data provides brand-specific corroboration. See §1 and
"Limitations" in the report.

## Layout

```
data/ab_inbev_us_revenue_growth.csv   quarterly US organic revenue growth, pre/post tagged
data/budlight_niq_weekly.csv          NIQ Bud Light off-premise weekly YoY
data/sources.csv                      every figure mapped to its published source
analysis.py                           statistics + figures; emits results.tex and results.json
report.tex                            the document; every number comes from results.tex
report.pdf                            built output
```

## Reproduce

```bash
pip install pandas numpy scipy matplotlib
python3 analysis.py                     # writes figures/, results.tex, results.json
pdflatex report.tex && pdflatex report.tex
```

No statistic in the PDF is typed by hand — `analysis.py` generates LaTeX macros
that `report.tex` consumes, so the prose cannot drift from the data.

## Method

- Classical z against the pre-period mean and SD.
- Standardised prediction residual `t = (x − x̄)/(s√(1+1/n))` on `n−1` df, the
  correct small-sample analogue when the baseline moments are themselves estimated.
- Robust z using median and 1.4826 × MAD.
- Welch t-test, Mann–Whitney U, and a 200k-draw permutation test on the split.
- Bootstrap percentile interval on z, resampling the baseline.
- Counterfactual revenue shortfall: baseline growth rate carried forward.
