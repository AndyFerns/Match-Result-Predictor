# FootballMatchPredictor

A Python library to train and predict soccer match outcomes using historical data and a Random Forest classifier.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Calculates team-level features (goals scored/conceded averages, form, points) over the last 5 matches
- Supports training on historical CSV datasets
- Uses `RandomForestClassifier` for prediction
- Simple API:
  - `train(...)` to build the model
  - `predict_match(home, away)` to get win/draw/lose probabilities

## Prerequisites

- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`

Install required packages:

```bash
pip install pandas numpy scikit-learn
```

## Installation

1. Clone this repository:
   ```bash
git clone https://github.com/your-username/FootballMatchPredictor.git
cd FootballMatchPredictor
```
2. (Optional) Create and activate a virtual environment:
   ```bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows
```
3. Install dependencies:
   ```bash
pip install -r requirements.txt
```

## Usage

1. Prepare a dataset CSV with columns:
   - `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`  
2. Update path in code or pass DataFrame/CSV path to `train()`.

```python
from football_predictor import FootballMatchPredictor

# Initialize for Premier League
predictor = FootballMatchPredictor('Premier League')
# Train on 2024-25 season data
predictor.train("Datasets/season-2425.csv")
# Predict a fixture
result = predictor.predict_match("Arsenal", "Liverpool")
print(result)
# Output: {'home_win': 0.65, 'draw': 0.20, 'away_win': 0.15}
```

## Project Structure

```
.
├── src/
│   └── football_predictor.py  # FootballMatchPredictor class
├── Datasets/
│   └── season-2425.csv        # Sample training data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration

- **Feature window**: currently uses last 5 matches for form/goals averages
- **Model**: Random Forest with 100 trees; adjust hyperparameters in `__init__`

## Contributing

Contributions are welcome! Please open issues or pull requests to:

1. Improve feature engineering
2. Add support for other leagues
3. Experiment with different models

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

