.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = loopqprize
PROFILE = babb56a80d22c9812ce3284396a3e94a61b1e970421e4a240ae8eccaffa74006
PROJECT_NAME = Loop Q Prize
PROJECT_NAME_VENV = loop_q_prize
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

ifeq (,bash -c pyenv)
HAS_PYENV=False
else
HAS_PYENV=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create folder structure needed
create_folder_structure:
	mkdir -p models/cnn_features;
	mkdir -p models/hog_features;
	mkdir -p models/pca_features;
	mkdir -p reports/figures;

requirements: 
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Download Data from S3
sync_data_from_s3: test_environment
	$(PYTHON_INTERPRETER) src/data/initialize_data.py

## Set up python interpreter environment
create_environment: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo "Install Pyenv dependencies..."
	sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
	libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
ifeq (False,$(HAS_PYENV))
	@echo ">>> Installing Pyenv-virtualenv"
	curl https://pyenv.run | bash
	@echo ">>> Installing python 3.6.10"
	@bash -c "pyenv install 3.6.10"
else
	@echo ">>> Pyenv is already installed."
endif
#	@echo ">>> Installing Pyenv python 3.6.10.."
#	@bash -c "pyenv install 3.6.10"
	@echo ">>> Initializing pyenv virtualenv"
	@bash -c "pyenv virtualenv 3.6.10 $(PROJECT_NAME_VENV)"
	@echo ">>> New virtualenv created"
	@bash -c "pyenv local $(PROJECT_NAME_VENV)"
endif

## Test python environment is setup correctly
test_environment: create_folder_structure
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
