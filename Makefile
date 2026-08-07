PYTHON ?= python

.PHONY: setup models data train evaluate optimize validate test dashboard all

setup:
	$(PYTHON) -m pip install -r requirements.txt

models:
	$(PYTHON) scripts/download_models.py

data:
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

optimize:
	$(PYTHON) scripts/optimize_model.py

validate:
	$(PYTHON) scripts/validate_agent_contracts.py
	$(PYTHON) scripts/validate_project.py

test:
	$(PYTHON) -m pytest

dashboard:
	$(PYTHON) -m streamlit run app/real_time_dashboard.py

all:
	$(PYTHON) scripts/run_pipeline.py
