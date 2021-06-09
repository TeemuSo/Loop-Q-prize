#!/bin/bash
echo "Install project dependencies.."
python3 -m pip install -r requirements.txt

echo "Install our src-package"
pip3 install -e .

echo "Test whether AWS connection is correctly configured..."
python3 src/tests/test_aws_connection.py

echo "Starting python script initialize_data.py"
python3 src/data/initialize_data.py

read -p "Press any key to resume ..."