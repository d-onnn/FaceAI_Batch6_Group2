# FaceAI Group 2 task runner. Run inside the activated env:  mamba activate faceai
#
# Every recipe is a single python call, so it runs the same under cmd.exe (what
# conda-forge make uses on Windows) and sh (macOS/Linux). Targets are added as each
# phase lands - only targets that work today are listed here.

PYTHON ?= python

.PHONY: help verify hardware probe lock test audit eda splits features cv diagnostics robustness clinical explain standards report03 report03b report03c phase3

help:
	@echo make verify   - environment verifier, webcam and face checks skipped
	@echo make hardware - detect hardware, write env/hardware.json
	@echo make probe    - Phase 0 dataset-access probe, write reports/00_access.md
	@echo make test     - pytest - formats, targets and integrity audit
	@echo make audit    - print the Phase 1 integrity checks
	@echo make eda      - Phase 1 audit + EDA, write reports/01_eda.md and figures
	@echo make splits   - Phase 2 subject-level splits, write artifacts/splits.parquet
	@echo make features - Phase 3 contact-PPG features: clean, noise, camera timing, length
	@echo make cv       - Phase 3 5-fold CV with bootstrap CIs, write artifacts/cv/
	@echo make diagnostics - Phase 3 overfitting curves: regularisation, boosting, learning
	@echo make robustness  - Phase 3 noise, shift, subgroups, stability, camera timing, gate
	@echo make clinical    - Phase 3b clinical thresholds and within-person change
	@echo make explain     - Phase 3b grouped permutation importance with fold stability
	@echo make standards   - Phase 3c errors judged by each accuracy standard in its own form
	@echo make report03 - Phase 3 report + figures from the saved artifacts
	@echo make report03b - Phase 3b report: gate, timing, length, clinical, change, why
	@echo make report03c - Phase 3c report: our errors against accuracy standards
	@echo make phase3   - all of the Phase 3 steps above, in order
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

features:
	$(PYTHON) -m faceai.features --noise --camera --duration

cv:
	$(PYTHON) -m faceai.cv

diagnostics:
	$(PYTHON) -m faceai.diagnostics

robustness:
	$(PYTHON) -m faceai.robustness

clinical:
	$(PYTHON) -m faceai.clinical

explain:
	$(PYTHON) -m faceai.explain

standards:
	$(PYTHON) -m faceai.standards

report03:
	$(PYTHON) -m faceai.report03

report03b:
	$(PYTHON) -m faceai.report03b

report03c:
	$(PYTHON) -m faceai.report03c

phase3: features cv diagnostics robustness clinical explain standards report03 report03b report03c

hardware:
	$(PYTHON) -m faceai.hardware

probe:
	$(PYTHON) -m faceai.data.probe

lock:
	$(PYTHON) -m faceai.lockfile
