#!/bin/bash
set -e

############################################################
# <<<<<<  CONFIGURABLE VARIABLES >>>>>>>>>>>>>>>>>>
############################################################

AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID="973759794702"
ECR_REPO_NAME="ml-inference-api"
ECR_URI="public.ecr.aws/v9s6o3o0/ml-inference-api"
EC2_SSH_KEY="~/Desktop/MLOPS/Project/mlops-project-key.pem
"
EC2_USER="ubuntu"   
SERVICE_PORT=8000
EC2_PUBLIC_IP= "16.171.132.23"

############################################################
# 
############################################################

IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo ""
echo "========================================================="
echo "STEP 1: Building Docker image"
echo "========================================================="

cd "$(dirname "$0")/../ml_system/api"
docker build -t ${ECR_REPO_NAME}:${IMAGE_TAG} .

echo ""
echo "========================================================="
echo "STEP 2: Logging into AWS ECR"
echo "========================================================="

aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo ""
echo "========================================================="
echo "STEP 3: Creating ECR repository if not exists"
echo "========================================================="

aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region $AWS_REGION >/dev/null 2>&1 || \
aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region $AWS_REGION >/dev/null

echo ""
echo "========================================================="
echo "STEP 4: Tagging and pushing image to ECR"
echo "========================================================="

docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${ECR_URI}
docker push ${ECR_URI}

echo ""
echo "========================================================="
echo "STEP 5: SSH into EC2 and deploy container"
echo "========================================================="

ssh -o StrictHostKeyChecking=no -i ${EC2_SSH_KEY} ${EC2_USER}@${EC2_PUBLIC_IP} << EOF

  set -e

  echo ""
  echo "Updating EC2 instance..."
  sudo apt-get update -y >/dev/null
  sudo apt-get install -y docker.io >/dev/null

  echo ""
  echo "Logging EC2 into AWS ECR..."
  aws ecr get-login-password --region ${AWS_REGION} \
    | sudo docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

  echo ""
  echo "Pulling latest Docker image..."
  sudo docker pull ${ECR_URI}

  echo ""
  echo "Stopping old container (if exists)..."
  sudo docker stop ml-api >/dev/null 2>&1 || true
  sudo docker rm ml-api >/dev/null 2>&1 || true

  echo ""
  echo "Running new container..."
  sudo docker run -d \
    --name ml-api \
    -p ${SERVICE_PORT}:8000 \
    --restart unless-stopped \
    ${ECR_URI}

  echo ""
  echo "Deployment complete on EC2."
  echo "Service running at: http://${EC2_PUBLIC_IP}:${SERVICE_PORT}"

EOF

echo ""
echo "========================================================="
echo "DEPLOYMENT FINISHED 🎉"
echo "Your ML Model API is LIVE at:"
echo "http://${EC2_PUBLIC_IP}:${SERVICE_PORT}"
echo "========================================================="
