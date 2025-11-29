#!/bin/bash
set -e

############################################################
# Config
############################################################

AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID="973759794702"
ECR_REPO_NAME="ml-inference-api"
EC2_SSH_KEY="~/Desktop/MLOPS/Project/mlops-project-key.pem"
EC2_PUBLIC_IP="16.171.132.23"
EC2_USER="ubuntu"
SERVICE_PORT="8000"

IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

############################################################
# Step 1 – Build Docker Image (from project root)
############################################################

echo ""
echo "========================================================="
echo "STEP 1: Building Docker Image"
echo "========================================================="

cd "$(dirname "$0")/.."      # <-- move to project root

docker build \
    -t ${ECR_REPO_NAME}:${IMAGE_TAG} \
    -f ml_system/api/Dockerfile .

############################################################
# Step 2 – Login to ECR
############################################################

echo ""
echo "========================================================="
echo "STEP 2: Login to AWS ECR"
echo "========================================================="

aws ecr get-login-password --region $AWS_REGION \
 | docker login \
     --username AWS \
     --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

############################################################
# Step 3 – Create ECR Repo if needed
############################################################

echo ""
echo "========================================================="
echo "STEP 3: Creating ECR Repo if not exists"
echo "========================================================="

aws ecr describe-repositories \
    --repository-names ${ECR_REPO_NAME} \
    --region $AWS_REGION >/dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name ${ECR_REPO_NAME} \
    --region $AWS_REGION >/dev/null

############################################################
# Step 4 – Push Image to ECR
############################################################

echo ""
echo "========================================================="
echo "STEP 4: Push Docker Image"
echo "========================================================="

docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${ECR_URI}
docker push ${ECR_URI}

############################################################
# Step 5 – SSH into EC2 and Deploy
############################################################

echo ""
echo "========================================================="
echo "STEP 5: Deploying to EC2"
echo "========================================================="

ssh -o StrictHostKeyChecking=no -i ${EC2_SSH_KEY} ${EC2_USER}@${EC2_PUBLIC_IP} << EOF
set -e

sudo apt-get update -y >/dev/null
sudo apt-get install -y docker.io awscli >/dev/null

aws ecr get-login-password --region ${AWS_REGION} \
 | sudo docker login \
     --username AWS \
     --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

sudo docker pull ${ECR_URI}

sudo docker stop ml-api >/dev/null 2>&1 || true
sudo docker rm ml-api >/dev/null 2>&1 || true

sudo docker run -d \
    --name ml-api \
    -p ${SERVICE_PORT}:8000 \
    --restart unless-stopped \
    ${ECR_URI}

EOF

echo ""
echo "========================================================="
echo "DEPLOYMENT COMPLETE 🎉"
echo "Your ML Model API is LIVE at:"
echo "http://${EC2_PUBLIC_IP}:${SERVICE_PORT}"
echo "========================================================="
