#!/bin/bash

# Check if the virtual environment exists, if not create one
if [ ! -d "myenv" ]; then
    python3 -m venv myenv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source myenv/bin/activate

# Install the requirements
pip install -r requirements.txt

# Apply migrations to set up the database
python manage.py migrate

# Create a superuser for Django admin if required
# Uncomment the following lines if needed
# python manage.py createsuperuser

# Run the server
python manage.py runserver
