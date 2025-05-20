#!/bin/bash
set -e

# Only process AWS credentials if USE_DVC is enabled
if [[ "${USE_DVC,,}" =~ ^(1|true|yes|y)$ ]]; then
  echo "USE_DVC is enabled, copying AWS credentials to the container..."
  export AWS_SHARED_CREDENTIALS_FILE=/trainer/.aws/credentials
  mkdir -p $(dirname $AWS_SHARED_CREDENTIALS_FILE)
  cp /run/secrets/aws_credentials $AWS_SHARED_CREDENTIALS_FILE
  chown trainer:dev $AWS_SHARED_CREDENTIALS_FILE
  chmod 600 $AWS_SHARED_CREDENTIALS_FILE
else
  echo "USE_DVC is not enabled, skipping AWS credentials setup"
fi

# Run pull_data.sh to prepare the datasets
echo "Preparing datasets..."
chmod +x ./scripts/pull_data.sh
bash ./scripts/pull_data.sh "$USE_DVC"

# Switch user
exec gosu trainer "$@"
