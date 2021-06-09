#!/bin/bash
echo "Test whether AWS connection is correctly configured..."
python src/tests/test_aws_connection.py


echo "Starting python script initialize_data.py"
#python src/data/initialize_data.py

read  -n 1 -p "Press any key to close the prompt.." mainmenuinput
