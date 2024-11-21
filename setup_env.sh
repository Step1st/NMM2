#!/bin/bash
mkdir animations
mkdir csv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt