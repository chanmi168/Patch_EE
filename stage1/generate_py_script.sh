#!/bin/bash

set -e
set -u

# jupyter nbconvert --to python read_cosmed.ipynb
jupyter nbconvert --to python read_patch.ipynb