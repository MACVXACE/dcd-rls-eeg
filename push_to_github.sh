#!/bin/zsh
# Publishes this folder as the public GitHub repository MACVXACE/dcd-rls-eeg.
# Run from inside the unzipped folder:  zsh push_to_github.sh
set -e
REPO="MACVXACE/dcd-rls-eeg"
URL="https://github.com/$REPO"

cd "$(dirname "$0")"

if [[ ! -f supplement.pdf ]]; then
  echo "ERROR: supplement.pdf is missing from $(pwd). Add it, then run this script again."
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed. Run: xcode-select --install"
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is not installed. Run: brew install gh   then: gh auth login"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is not logged in. Run: gh auth login"
  exit 1
fi
if [[ -z "$(git config --global user.name)" || -z "$(git config --global user.email)" ]]; then
  echo "ERROR: git name/email not set. Run:"
  echo '  git config --global user.name "Varun Bajpai"'
  echo '  git config --global user.email "varunku3117p@gmail.com"'
  exit 1
fi
if [[ -d .git ]]; then
  echo "ERROR: this folder is already a git repository. Remove the .git folder or push manually."
  exit 1
fi

git init -b main
git add -A
echo "---- git status ----"
git status
git commit -m "Code, output logs and paper sources for the fixed-point ridge DCD-RLS EEG preprint"
gh repo create "$REPO" --public --source=. --remote=origin --push
echo ""
echo "Live at: $URL"
