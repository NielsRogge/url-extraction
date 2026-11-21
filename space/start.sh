#!/bin/bash
# Move the HF token out of the environment before anything else starts: the agents run shell
# commands in this container and must not be able to read it from /proc/*/environ.
set -euo pipefail
umask 077
if [ -n "${HF_TOKEN:-}" ]; then printf '%s' "$HF_TOKEN" > /tmp/.hf_token; fi
exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN python launch.py
