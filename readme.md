# WaveConf — Bitcoin Elliott Wave Analysis System

> This project is inspired by the analytical framework employed by **Asst. Prof. Dr. Gema Goeyardi, CAT,CFTe,MFTA founder of ASTRONACCI TIME TRADING which combines rule-based Elliott Wave detection, Fibonacci**, which combines rule-based Elliott Wave detection, Fibonacci confluence analysis, and planetary, lunar, and macroeconomic cycle indicators. The objective is to model and augment this analytical process using a Temporal Fusion Transformer (TFT), deployed on a home Linux server with six-hour automated updates and Telegram integration for real-time notifications.

---

## What This Is

WaveConf is a personal research project that automates a specific multi-layer confluence trading methodology:

1. Run **two independent structural analyses** on the same price action
2. Project a price target from each independently using Fibonacci extensions
3. Only treat a zone as actionable when **both targets cluster within 2%** of each other
4. Cross-validate the classical signal with a **TFT deep learning model** trained on 11 years of BTC daily data
5. Overlay **planetary cycle features** and **US economic calendar events** as behavioral context

The system is not a generic trading bot. It encodes the specific decision process of one analyst — with the goal that the AI produces the same structural reads a trained human would, narrated every 6 and 12 hours via Telegram and a live HTML dashboard served on your home network.

---

## Architecture

```
Data Ingestion (CCXT / Binance)
        │
        ▼
Dynamic Volatility ZigZag
  Two-layer pivot detection
  (macro: institutional swings │ micro: sub-wave pivots)
        │
        ▼
Structure Tokenizer
  HH / HL / LH / LL / BOS / CHOCH / DIV_H / DIV_L / FIB_T / SWEEP
        │
        ├──────────────────────────┐
        ▼                          ▼
  Track 1                     Track 2
  Geometric pattern            Pure Elliott re-count
  (ABW / wedge / channel)      (Flat 3-3-5 / Zigzag /
  + trendline fit               Triangle / Diagonal / Combo)
        │                          │
        └──────────┬───────────────┘
                   ▼
          Fibonacci Engine
          Dual-tool cluster:
          Tool A → 2.618 ext from C top
          Tool B → 1.618 ext from B→C range
          Only fires when both land within 2%
                   │
                   ▼
       Temporal Fusion Transformer
       Known future channel:
         • Lunar phase / Bradley score / Mercury Rx
         • FOMC / CPI / NFP calendar dates
       Outputs: q10 / q50 / q90 at t+7/14/30/60 days
                   │
                   ▼
         Confluence Scorer
         TFT q50 vs Fibonacci cluster alignment
         Calendar risk adjustment
                   │
                   ▼
        Telegram alert + Live Web Dashboard
       (every 6h invalidation check, 12h full update)
```

---

## Key Technical Decisions

**Pipeline, not monolith.** Math stays math — Fibonacci extensions are computed directly, not approximated by a neural network. The TFT handles sequence modeling and probabilistic forecasting. The two never cross lanes.

**Two independent analytical tracks.** The same price swing is analyzed simultaneously using geometric pattern detection (Track 1) and pure Elliott Wave re-count (Track 2). A signal only fires when both tracks point to the same price zone. Either track can independently invalidate a setup.

**Known-future TFT input channel.** Planetary positions are computable for any future date from orbital mechanics. FOMC dates are published 12 months in advance. Both feed the TFT's `time_varying_known_reals` channel rather than the observed-past channel — giving the model legitimate forward information rather than treating these as lagged features.

**Negative-target detection.** Linear Fibonacci extensions on large BTC drawdowns (>38.2% of price) produce mathematically negative targets. Every extension is validated before any cluster check runs. Log-scale extension is on the roadmap as a structural fix for this category.

**Astro features are unsigned.** The engine computes raw, continuous aspect intensity (0 at orb edge, 1.0 at exact angle) with no hardcoded bullish/bearish polarity. The TFT learns whatever directional weight the historical data supports — or assigns zero weight if there is none.

---

## Module Map

```
src/waveconf/
│
├── ingestion/
│   ├── fetch_ohlcv.py          CCXT/Binance data fetch, incremental SQLite cache
│   ├── calculate_layers.py     ATR-based dynamic volatility threshold calculation
│   ├── indicators.py           RSI, MACD, ATR, Bollinger width, price normalization
│   └── investing_api.py        Investing.com API integration to fetch macro events
│
├── pivots/
│   ├── pivot_schema.py         PivotPoint dataclass, SwingType, StructureLabel, WaveDegree
│   ├── zigzag.py               Two-layer state machine ZigZag (macro + micro)
│   ├── classifiers.py          ImpulseClassifier + CorrectionClassifier (15 wave types)
│   └── pattern_detector.py     Geometric pattern recognition from trendline pair
│
├── structure/
│   └── structure_tokenizer.py  HH/HL/LH/LL/BOS/CHOCH/DIV/FIB_T/SWEEP token emission
│
├── fib_engine/
│   ├── fibonacci.py            FibonacciEngine: extensions, retracements, dual_cluster
│   ├── trendline.py            Trendline dataclass + OLS fit_trendline()
│   ├── invalidation.py         Real-time price invalidation checks
│   └── measured_move.py        Pattern measured-move target computation
│
├── confluence/
│   ├── cluster_check.py        2% proximity cluster validation
│   ├── scorer.py               Confluence strength scoring + calendar risk adjustment
│   ├── entry_plan.py           Scenario A/B entry sizing plan
│   └── multi_tf.py             Cross-timeframe confluence correlation
│
├── wave_model/
│   ├── astro_features.py       PySwisseph planetary features (unsigned, continuous)
│   ├── dataset.py              DatasetBuilder: full pipeline → TFT-ready DataFrame
│   ├── model.py                Temporal Fusion Transformer configuration
│   ├── train.py                PyTorch Lightning training loop
│   └── infer.py                Frozen .pt inference for home server deployment
│
└── pipeline.py                 Orchestrates the full 9-step analytical process

scripts/
├── run_daily_analysis.py       Main daily analysis pipeline runner
├── economic_notifier.py        Standalone economic calendar alert engine
├── visualize_backtest.py       Generates the HTML dashboard visualization
├── setup_server.sh             Ubuntu server environment setup script
├── elliott-web.service         Systemd web server service configuration
├── elliott-notifier.service    Systemd notifier service configuration
└── elliott-notifier.timer      Systemd notifier 5-minute timer configuration
```

---

## Wave Type Coverage

**Impulse structures (6 types)**
Standard impulse · Wave 3 extension · Wave 5 extension · Wave 1 extension · Leading diagonal · Ending diagonal

**Correction structures (9 types)**
Regular flat (3-3-5) · Expanded flat · Running flat · Single zigzag (5-3-5) · Double zigzag · Triple zigzag · Contracting symmetrical triangle · Ascending triangle · Descending triangle · Double three combination · Triple three combination

All types are formally specified in `config/correction_rules.yaml` with per-type Fibonacci ratio bounds, B-breach flags, and diagonal overlap exceptions.

---

## Backtest Summary (Historical BTC, 2015–2026)

Confluence signals fire on approximately **9–15% of macro pivot events**, which is the expected behavior — the 2% proximity threshold is designed to be selective.

| Timeframe | Signals (11 years) | Win Rate | Avg Favorable Excursion | Avg Adverse Excursion |
|---|---|---|---|---|
| 1D Daily | 10 | 44.4% | 56.6% | 0.6% |
| 4H | 102 | 29.4% | 15.3% | 1.5% |

The asymmetry matters more than the win rate. On 1D: average winning move is 56.6%, average losing move is 0.6% — a ~94:1 favorable excursion ratio. Even at a 44% win rate, the expected value per signal is strongly positive. The tight 0.5% invalidation buffer is the mechanism that keeps the losing moves small.

---

## Live Output Format

```
📈 BTC ELLIOTT WAVE FORECAST [1D]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Current price: $62,436
📐 Bias: BULLISH — projecting UP from confirmed low

⏳ No confluence zone confirmed
   Target A: $101,958 | Target B: $81,461 (gap: 20.2%, above 2% threshold)

📊 TFT QUANTILE FORECAST
  t+ 7d  q10=$58,896  q50=$62,964  q90=$67,312
  t+14d  q10=$58,275  q50=$64,114  q90=$70,537
  t+30d  q10=$56,488  q50=$64,132  q90=$72,810
  t+60d  q10=$51,960  q50=$62,338  q90=$74,788

🌙 Lunar: 287° (waning gibbous)   ♄ Bradley: -0.42
📅 FOMC: 8 days   NFP: 3 days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Next update: 12:00 UTC
```

Non-confluence updates are sent every 6/12 hours regardless — the system narrates the current wave position, TFT quantile drift, and why no cluster fired. Silence is not an output.

---

## Stack

| Component | Technology |
|---|---|
| Data fetch | `ccxt` (Binance) |
| OHLCV cache | SQLite + Parquet |
| Indicators | Pure pandas/numpy |
| Pivot detection | Custom two-layer ZigZag state machine |
| Wave classification | Rule-based (1,245 lines, 15 wave types) |
| Deep learning model | Temporal Fusion Transformer via `pytorch-forecasting` + PyTorch Lightning |
| Planetary features | `pyswisseph` (Swiss Ephemeris) |
| Scheduling | APScheduler |
| Alerts | Telegram Bot API |
| Dashboard | Python http.server + Plotly (HTML) |
| Testing | pytest — 105 tests |
| Deployment | Home Linux server, 8 GB RAM, inference-only (model trained externally on Colab/Kaggle) |

---

## Setup

```bash
git clone https://github.com/rezahanif/ElliottWaveAnalyzer.git
cd ElliottWaveAnalyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pyswisseph pytorch-forecasting pytorch-lightning

# Fetch BTC data and compute volatility layers
python -m src.waveconf.ingestion.fetch_ohlcv
python -m src.waveconf.ingestion.calculate_layers

# Run the full test suite
PYTHONPATH=. pytest tests/

# Run daily analysis (dry run)
PYTHONPATH=. python scripts/run_daily_analysis.py --dry-run
```

A trained `wave_model.pt` is required for TFT inference. Train on your own hardware or Colab using `src/waveconf/wave_model/train.py`, then place the weights at `models/wave_model.pt`. The home server loads weights at startup and hot-reloads when the file is updated.

---

## Production Deployment (Home Server)

To deploy the automated pipeline and monitoring dashboards on your home server:

1. **Setup the Web Dashboard Server (Port 8080)**:
   ```bash
   sudo cp scripts/elliott-web.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now elliott-web.service
   ```
   Access the dashboard locally at `http://<YOUR-SERVER-IP>:8080/confluence_visualization.html`.

2. **Setup the Economic Event Notifier (Runs every 5 minutes)**:
   ```bash
   sudo cp scripts/elliott-notifier.service /etc/systemd/system/
   sudo cp scripts/elliott-notifier.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now elliott-notifier.timer
   ```

---

## Known Limitations

**Log-scale projections not yet implemented.** Linear Fibonacci extensions on BTC drawdowns exceeding ~38.2% of price produce negative targets, which are automatically filtered. This suppresses valid bearish signals on the 1D timeframe during large bear markets. Log-scale extension is the planned fix.

**Weekly timeframe limited by pivot count.** 9 macro pivots on 1W is insufficient for reliable confluence computation. The system currently emits TFT quantile forecasts on the weekly channel without a Fibonacci cluster.

**Astro features are unvalidated priors.** The planetary cycle weights in `config/astro_features.yaml` are initial priors, not backtested values. The TFT will learn the empirical weights from data during training — but treating the config values as validated is incorrect until a proper backtest of realized volatility around each event type is run.

**Training data size.** ~4,300 daily candles (11 years BTC) is a small dataset by ML standards. The TFT is sized accordingly (hidden_size=32, attention_head_size=2) to avoid overfitting, but generalization to significantly different market regimes is uncertain.

---

## Project Status

Core pipeline complete and tested. Live on home server producing 6h/12h Telegram updates. Active work areas: log-scale Fibonacci extension, wave projection engine (narrating in-progress wave formation rather than only completed signals), and confluence confidence recalibration via historical FOMC volatility analysis.

---

*Personal research project. Not financial advice. All signals are probabilistic structural reads, not guaranteed outcomes.*
