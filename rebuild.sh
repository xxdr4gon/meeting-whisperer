#!/bin/bash

echo "Rebuilding containers with pre-downloaded Estonian model..."
echo "This will take several minutes as it downloads the model during build."

# Stop existing containers
docker compose down

# Remove old images to force rebuild
docker rmi meeting-whisperer-app meeting-whisperer-worker 2>/dev/null || true

# Rebuild and start
docker compose up --build -d

echo "Build complete! The Estonian model should now be pre-downloaded."
echo "Check logs with: docker compose logs -f worker"
