Steel Price Predictor (Ensemble Model V3)
Why this project?
As a Civil Engineering student at Zagazig University, I’ve seen firsthand how the volatility of building material prices—especially steel—can disrupt project budgets and construction timelines in Egypt. I wanted to bridge the gap between my engineering domain and Data Science to create a tool that helps stakeholders anticipate market shifts rather than just reacting to them.

How it Works
This isn't just a simple linear regression. To handle the complexity of the Egyptian market (Inflation, USD rates, and global trends), I built a Hybrid Ensemble Model:

Random Forest: For capturing non-linear relationships in economic data.

XGBoost: For high-performance gradient boosting to minimize errors.

Weighting: A 50/50 split between both models to achieve a balanced, more stable forecast (MAPE ~4.5%).

Key Indicators Tracked
The model looks at what actually moves the price in the real world:

Macroeconomics: USD/EGP exchange rates and local Inflation.

Global Market: Iron Ore prices and Energy costs (Brent Oil & Natural Gas).

Market Lag: I factored in "Lagged Indicators" because price changes in global raw materials usually take a few weeks to hit the local retail market.

Tech Stack
Python: The backbone of the logic.

Streamlit: To turn the code into an interactive dashboard for non-technical users.

Joblib: For efficient model serialization.

Getting Started
Clone the repo.

Install requirements: pip install -r requirements.txt.

Run the app: streamlit run constructionapp.py.

Future Work
Integrating a real-time API to fetch USD and Iron Ore prices automatically.

Adding more materials like Cement and Aluminum.

Ibrahim Mohamed
Civil Engineering & AI Enthusiast
