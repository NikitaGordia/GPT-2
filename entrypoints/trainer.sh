#!/bin/sh
set -e

echo "Copying AWS credentials to the container..."
export AWS_SHARED_CREDENTIALS_FILE=/trainer/.aws/credentials
mkdir -p $(dirname $AWS_SHARED_CREDENTIALS_FILE)
cp /run/secrets/aws_credentials $AWS_SHARED_CREDENTIALS_FILE
chown trainer:dev $AWS_SHARED_CREDENTIALS_FILE
chmod 600 $AWS_SHARED_CREDENTIALS_FILE

# Switch user
exec gosu trainer "$@"
