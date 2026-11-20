FROM vllm/vllm-openai:v0.33.0

# App environment, separate from vLLM's so their dependency pins never conflict.
RUN pip install --no-cache-dir uv \
 && uv venv /opt/app --python "$(command -v python3)" \
 && uv pip install --python /opt/app/bin/python --no-cache \
      fastapi "uvicorn[standard]" httpx starlette "opencode-ai==0.1.0a36" "pydantic>=2.10,<2.13" \
      "datasets>=4.0.0" "huggingface_hub==2.4.0" "click==8.5.1" \
 && curl -fsSL https://github.com/anomalyco/opencode/releases/download/v1.2.4/opencode-linux-x64.tar.gz \
      | tar -xz -C /opt/app/bin opencode \
 && chmod +x /opt/app/bin/opencode

# Spaces run the container as UID 1000 (the base image already has a user with that UID).
RUN mkdir -p /home/user/app && chown -R 1000:1000 /home/user
USER 1000
ENV HOME=/home/user \
    PATH=/opt/app/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=1000:1000 . .

EXPOSE 7860
ENTRYPOINT []
CMD ["bash", "start.sh"]
