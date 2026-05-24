# Hack4Health - CVD Detection

Streamlit Cloud-ready version of the cardiovascular disease risk predictor.

## Tuning

Run the hyperparameter search with:

```bash
python tuning.py
```

Outputs are written to `outputs/best_model.pkl`, `outputs/tuning_results.csv`, and `outputs/tuning_plot.png`.

## Streamlit Cloud

Use `app.py` as the main file and `requirements.txt` as the dependency list.

Deploy steps:

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Choose the repository and set the main file path to `app.py`.
4. Deploy.

## Local run

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
