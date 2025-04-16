#!/bin/bash
set -e

AWS_SECRET_FILE="/run/secrets/aws_credentials"
if [ -f "$AWS_SECRET_FILE" ]; then
  echo "Configuring AWS credentials from Docker secret..."
  export AWS_SHARED_CREDENTIALS_FILE="$AWS_SECRET_FILE"
fi

echo "Pulling data with DVC..."

dvc pull

echo "DVC pull completed."

echo "Executing command: $@"
exec "$@"