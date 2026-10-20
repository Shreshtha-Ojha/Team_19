#!/usr/bin/env bash
# Deploy the web app to a Hugging Face Space (free hosting).
#
# One-time setup:
#   1. Create an account at https://huggingface.co
#   2. Create a Space at https://huggingface.co/new-space
#      - SDK: Docker, template: Blank, hardware: CPU basic (free)
#   3. Create an access token with "write" permission at
#      https://huggingface.co/settings/tokens
#
# Then, from the repository root:
#   scripts/deploy_hf.sh <hf-username>/<space-name>
#
# When git asks for a username/password, enter your Hugging Face username
# and paste the token as the password.
#
# The Space only receives what the app needs (src/, requirements.txt,
# Dockerfile) plus a README with the settings Hugging Face requires.
# Run the script again after changing the code to update the Space.
set -euo pipefail

SPACE="${1:?usage: scripts/deploy_hf.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# SPACE_URL can point elsewhere for testing.
git clone --depth 1 "${SPACE_URL:-https://huggingface.co/spaces/$SPACE}" "$TMP/space"
cd "$TMP/space"

# Replace everything except git metadata with the current app files.
find . -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitattributes -exec rm -rf {} +
cp -r "$ROOT/src" "$ROOT/Dockerfile" "$ROOT/requirements.txt" .
find src -name __pycache__ -prune -exec rm -rf {} +

cat > README.md <<README
---
title: Low-Light Enhancer
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Classical low-light photo enhancement, step by step
---

Enhances dark photos with classical image processing only: no AI, no training.

Source code and documentation: https://github.com/Shreshtha-Ojha/lowlight-enhancement
(deployed from commit $COMMIT).
README

git add -A
if git diff --cached --quiet; then
    echo "Space is already up to date."
    exit 0
fi
git commit -q -m "Deploy $COMMIT from GitHub"
git push
echo
echo "Pushed. Hugging Face is now building it (a few minutes):"
echo "  https://huggingface.co/spaces/$SPACE"
