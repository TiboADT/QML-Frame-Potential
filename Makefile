VENV_DIR=.venv

.PHONY: venv install activate

venv:
	python3 -m venv $(VENV_DIR)

install: .venv
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	$(VENV_DIR)/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

activate:
	@echo "Run: source $(VENV_DIR)/bin/activate"

reset:
	rm -rf $(VENV_DIR)
	$(MAKE) install
