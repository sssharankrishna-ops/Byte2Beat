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

## App authentication

The dashboard now uses a login gate before loading the model or charts. Configure credentials with Streamlit secrets or environment variables.

### Streamlit secrets

Create `.streamlit/secrets.toml` locally or add the same values in Streamlit Cloud secrets:

```toml
auth_username = "your-username"
auth_password = "your-strong-password"
```

### Environment variables

```bash
APP_AUTH_USERNAME=your-username
APP_AUTH_PASSWORD=your-strong-password
```

This is application-level access control. A true firewall is handled by the hosting platform, not by `app.py` itself.

## Local run

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
