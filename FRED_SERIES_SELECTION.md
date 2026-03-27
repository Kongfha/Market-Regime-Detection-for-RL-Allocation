# FRED Series Selection

This note summarizes the recommended FRED macro panel for the portfolio management project.

## Recommendation

- Main modeling panel: `16` raw FRED series
- Sensitivity / ablation panel: up to `23` raw FRED series
- Do not expand the main panel much beyond `18-20` raw series unless you are committed to PCA before HMM fitting

Why:

- The weekly training window is limited, so adding too many macro series creates redundancy and makes the regime model harder to estimate reliably.
- The goal is not to maximize series count. The goal is to cover the main macro-financial themes with minimal overlap.

## Coverage Themes

The macro panel should cover these themes:

1. Policy and front-end rates
2. Long-end rates and yield curve slope
3. Inflation expectations
4. Broad financial conditions and credit stress
5. Market risk and dollar tightness
6. Liquidity
7. Labor
8. Real activity
9. Consumer and housing

## Presets

The data fetch script now supports named presets:

```bash
python3 scripts/fetch_fred_macro_panel.py --preset compact
python3 scripts/fetch_fred_macro_panel.py --preset core
python3 scripts/fetch_fred_macro_panel.py --preset extended
```

## Compact Panel (10 series)

This is the original lightweight panel:

- `DFF`
- `DGS10`
- `T10Y2Y`
- `T10YIE`
- `NFCI`
- `ICSA`
- `CPIAUCSL`
- `UNRATE`
- `INDPRO`
- `UMCSENT`

## Core Panel (16 series)

This is the recommended main panel for the project.

- `DFF`: policy stance
- `DGS3MO`: front-end rate
- `DGS10`: long-end rate
- `T10Y3M`: curve slope / inversion
- `T10YIE`: inflation expectations
- `NFCI`: broad financial conditions
- `BAMLH0A0HYM2`: high-yield credit spread
- `VIXCLS`: market risk / implied volatility
- `DTWEXBGS`: broad dollar index
- `WRESBAL`: banking-system liquidity
- `ICSA`: weekly labor stress
- `UNRATE`: labor confirmation
- `CPIAUCSL`: realized inflation
- `CFNAI`: broad activity composite
- `UMCSENT`: consumer sentiment
- `PERMIT`: housing leading indicator

## Extended Panel (23 series)

Use this for sensitivity checks, not as the default first model.

- `DFF`
- `DGS3MO`
- `DGS10`
- `T10Y3M`
- `T10YIE`
- `T5YIFR`
- `NFCI`
- `ANFCI`
- `STLFSI4`
- `BAMLH0A0HYM2`
- `BAMLC0A4CBBB`
- `VIXCLS`
- `DTWEXBGS`
- `WRESBAL`
- `ICSA`
- `CC4WSA`
- `CPIAUCSL`
- `UNRATE`
- `PAYEMS`
- `CFNAI`
- `UMCSENT`
- `PERMIT`
- `MORTGAGE30US`

## Series To Avoid In The Main Panel

These are useful, but not ideal for the main historical panel:

- `SOFR`: starts in 2018, which shortens the training history
- `IORB`: starts in 2021, which shortens the training history even more

## Redundancy Rules

- Choose only one main curve spread in the first-pass model
- Choose only one main broad financial conditions index in the first-pass model
- Avoid stacking too many labor series together
- Avoid stacking too many credit spreads together
- Use the extended panel only after the core panel is stable

## Suggested Workflow

1. Start with `--preset core`
2. Resample and lag macro features causally
3. Standardize on train only
4. Apply PCA before HMM fitting if the combined state vector gets too wide
5. Compare `price-only` vs `price + core macro`
6. Use `--preset extended` only for robustness checks

## Official FRED Links

- [FRED API Overview](https://fred.stlouisfed.org/docs/api/fred/overview.html)
- [DFF](https://fred.stlouisfed.org/series/DFF)
- [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)
- [DGS10](https://fred.stlouisfed.org/series/DGS10)
- [T10Y3M](https://fred.stlouisfed.org/series/T10Y3M)
- [T10YIE](https://fred.stlouisfed.org/series/T10YIE)
- [NFCI](https://fred.stlouisfed.org/series/NFCI)
- [BAMLH0A0HYM2](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
- [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS)
- [DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS)
- [WRESBAL](https://fred.stlouisfed.org/series/WRESBAL)
- [ICSA](https://fred.stlouisfed.org/series/ICSA)
- [UNRATE](https://fred.stlouisfed.org/series/UNRATE)
- [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL)
- [CFNAI](https://fred.stlouisfed.org/series/CFNAI)
- [UMCSENT](https://fred.stlouisfed.org/series/UMCSENT)
- [PERMIT](https://fred.stlouisfed.org/series/PERMIT)
