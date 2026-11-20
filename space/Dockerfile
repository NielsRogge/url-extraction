FROM vllm/vllm-openai:v0.33.0

# App environment, separate from vLLM's so their dependency pins never conflict.
RUN pip install --no-cache-dir uv \
 && uv venv /opt/app --python "$(command -v python3)" \
 && uv pip install --python /opt/app/bin/python --no-cache \
      fastapi "uvicorn[standard]" httpx starlette "opencode-ai==0.1.0a36" "pydantic>=2.10,<2.13" \
      "datasets>=4.0.0" "huggingface_hub==2.4.0" "click==8.5.1" \
 && curl -fsSL https://github.com/anomalyco/opencode/releases/download/v1.2.4/opencode-linux-x64.tar.gz \
      | tar -xz -C /opt/app/bin opencode \
 && chmod +x /opt/app/bin/opencode \
 && ln -sf "$(command -v vllm)" /usr/local/bin/vllm || true

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/opt/app/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user . .

EXPOSE 7860
ENTRYPOINT []
CMD ["bash", "start.sh"]
