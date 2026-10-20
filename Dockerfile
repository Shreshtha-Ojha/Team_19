# Recipe a hosting service (e.g. Hugging Face Spaces) follows to build and run
# the web app. Not needed for running it on your own computer.
FROM python:3.12-slim

# Hugging Face Spaces run containers as user 1000; match that.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
WORKDIR /home/user/app/src
EXPOSE 7860
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "7860"]
