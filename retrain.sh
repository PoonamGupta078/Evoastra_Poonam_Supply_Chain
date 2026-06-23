#!/bin/bash
set -e

echo "================================================"
echo " Supply Chain ML — Retraining Pipeline"
echo "================================================"

echo ""
echo "[1/5] Checking for data drift..."
# We allow this to fail to capture the exit code
set +e
python src/drift_detector.py
DRIFT_EXIT=$?
set -e

if [ $DRIFT_EXIT -eq 1 ]; then
    echo "WARNING: Major drift detected. Retraining is strongly recommended."
else
    echo "INFO: No major drift detected. Proceeding with scheduled retrain."
fi

echo ""
echo "[2/5] Training regression model..."
python src/train.py

echo ""
echo "[3/5] Training classifier model..."
python src/train_classifier.py

echo ""
echo "[4/5] Training forecast model..."
python src/forecast.py

echo ""
echo "[5/5] Running evaluation..."
python src/evaluate.py

echo ""
echo "================================================"
echo " All models retrained successfully."
echo " Restart Docker container to serve new models:"
echo " docker compose -f deployment/docker-compose.yml restart"
echo "================================================"
