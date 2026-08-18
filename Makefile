PYTHON ?= python

.PHONY: setup models data preflight train calibrate freeze verify evaluate optimize validate test dashboard all

setup:
	$(PYTHON) -m pip install -r requirements.txt

models:
	$(PYTHON) scripts/download_models.py

data:
	$(PYTHON) scripts/prepare_data.py

preflight:
	$(PYTHON) scripts/preflight_revision2.py

train:
	$(PYTHON) scripts/train_revision2.py

calibrate:
	$(PYTHON) scripts/calibrate_validation.py

freeze:
	$(PYTHON) scripts/freeze_production.py

verify:
	$(PYTHON) scripts/verify_production_validation.py

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
