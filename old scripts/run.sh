#!/bin/bash
# Convenience script to activate the uv environment and run marimo

source .venv/bin/activate

if [ "$1" = "run" ]; then
    marimo run climate_models_blog.py
elif [ "$1" = "export" ]; then
    marimo export script climate_models_blog.py
elif [ "$1" = "edit" ]; then
    marimo edit climate_models_blog.py
else
    marimo run climate_models_blog.py
fi
