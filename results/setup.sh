#!/bin/bash
# Tworzy lokalne środowisko Python i instaluje potrzebne pakiety
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv_mac"

if [ ! -d "$VENV_DIR" ]; then
    echo "Tworzę venv w $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install mistralai python-dotenv -q

echo ""
echo "Gotowe. Środowisko: $VENV_DIR"
echo ""
echo "Uruchom skrypty tak:"
echo "  source $VENV_DIR/bin/activate"
echo "  python $SCRIPT_DIR/generate_base_mistral.py   # krok 1: generowanie"
echo "  python $SCRIPT_DIR/score_base_mistral.py      # krok 2: ocenianie"
