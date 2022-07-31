#!/bin/sh

poetry install
poetry run tensorboard --logdir ./outputs --bind_all
/bin/bash