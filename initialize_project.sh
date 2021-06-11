#!/bin/bash
echo "Install project dependencies.."
$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

echo "Install our src-package"
pip3 install -e .

echo "Test whether AWS connection is correctly configured..."
$(PYTHON_INTERPRETER) test_environment.py

echo "Starting python script initialize_data.py"
$(PYTHON_INTERPRETER) src/data/initialize_data.py

read -p "Press any key to resume ..."