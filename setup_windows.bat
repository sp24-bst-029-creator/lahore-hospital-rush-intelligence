@echo off
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_synthetic_data.py
python data_processing.py
python train_model.py
streamlit run app.py
pause
