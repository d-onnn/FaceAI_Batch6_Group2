# FaceAI Group 2 task runner. Run inside the activated env:  mamba activate faceai
#
# Every recipe is a single python call, so it runs the same under cmd.exe (what
# conda-forge make uses on Windows) and sh (macOS/Linux). Targets are added as each
# phase lands - only targets that work today are listed here.

PYTHON ?= python

.PHONY: help verify hardware probe lock test audit eda splits

help:
	@echo make verify   - environment verifier, webcam and face checks skipped
	@echo make hardware - detect hardware, write env/hardware.json
	@echo make probe    - Phase 0 dataset-access probe, write reports/00_access.md
	@echo make test     - pytest - formats, targets and integrity audit
	@echo make audit    - print the Phase 1 integrity checks
	@echo make eda      - Phase 1 audit + EDA, write reports/01_eda.md and figures
	@echo make splits   - Phase 2 subject-level splits, write artifacts/splits.parquet
	@echo make lock     - re-record exact installed versions - run after EVERY install

verify:
	$(PYTHON) tests/verify_env.py --no-camera

test:
	$(PYTHON) -m pytest tests/ -q --ignore=tests/verify_env.py

audit:
	$(PYTHON) -m faceai.data.audit

eda:
	$(PYTHON) -m faceai.eda

splits:
	$(PYTHON) -m faceai.splits

hardware:
	$(PYTHON) -m faceai.hardware

probe:
	$(PYTHON) -m faceai.data.probe

lock:
	$(PYTHON) -m faceai.lockfile
