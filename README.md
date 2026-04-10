# IoT Emergency System — Model Testing UI

PyQt5-based desktop application for batch-testing IoT detection models:
**Smoke / Fire**, **Water Leak**, **Gas Leak** (cumulants), and **Intrusion** (rule-based).

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/EgorovaViktoria/IoT.git
cd IoT

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the UI

```bash
python app.py
```

## Tabs

| Tab | Algorithm | Default dataset | Default model |
|-----|-----------|-----------------|---------------|
| 🔥 Smoke / Fire | Sklearn (pkl) + rolling window | `test/unified_test_clean.csv` | `models/smoke_model.pkl` |
| 💧 Water Leak | Sklearn (pkl) + rolling window | `test/unified_test_clean.csv` | `models/leak_model.pkl` |
| ⚗️ Gas Leak | Statistical cumulants (no pkl) | `test/unified_test_clean.csv` | — |
| 🚪 Intrusion | Rule-based (no pkl) | custom sensor-vector CSV | — |

### Intrusion tab

The intrusion algorithm uses rule-based logic (see `utilits/intrusion_detection.py`).
It requires a CSV with sensor vectors grouped by `batch_id`.  Click
**"Create sample CSV"** in the tab to generate an example file.

## Output

Results are saved to `test_results/`:

| File | Contents |
|------|----------|
| `test_results/fire.csv` | Smoke model predictions |
| `test_results/leak.csv` | Leak model predictions |
| `test_results/gas.csv`  | Gas detection predictions |
| `test_results/intrusion.csv` | Intrusion detection predictions |

## Project structure

```
app.py                  ← Entry point
requirements.txt
iot/
  ui/
    inference.py        ← Shared windowing / evaluation logic
    base_tab.py         ← Reusable tab widget base class
    smoke_tab.py        ← Smoke / fire tab
    leak_tab.py         ← Water leak tab
    gas_tab.py          ← Gas leak tab
    intrusion_tab.py    ← Intrusion detection tab
    main_window.py      ← Main window (QTabWidget)
models/
  smoke_model.pkl
  leak_model.pkl
test/
  unified_test_clean.csv
utilits/
  fire_test.py          ← Original CLI script (smoke)
  water_test.py         ← Original CLI script (leak)
  intrusion_detection.py
  cumulants.py
```
