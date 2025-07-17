#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

CLUSTER_NAME="multi-agent-bot-cluster"
K8S_DIR="k8s"

echo "--- Checking for existing KinD cluster: ${CLUSTER_NAME} ---"
if kind get clusters | grep -q "${CLUSTER_NAME}"; then
  echo "Existing cluster '${CLUSTER_NAME}' found. Deleting it..."
  kind delete cluster --name "${CLUSTER_NAME}"
  echo "Cluster '${CLUSTER_NAME}' deleted successfully."
else
  echo "No existing cluster '${CLUSTER_NAME}' found. Proceeding with creation."
fi

echo "--- Creating KinD cluster: ${CLUSTER_NAME} ---"
kind create cluster --config ${K8S_DIR}/kind-cluster.yaml --name ${CLUSTER_NAME}

echo "--- Loading Docker images into KinD cluster ---"
kind load docker-image base-mcp:latest --name ${CLUSTER_NAME}
kind load docker-image multi-agent-bot-api:latest --name ${CLUSTER_NAME}
kind load docker-image fastmcp-core-server:latest --name ${CLUSTER_NAME}
kind load docker-image telegram-mcp-server:latest --name ${CLUSTER_NAME}
kind load docker-image discord-mcp-server:latest --name ${CLUSTER_NAME}
kind load docker-image web-mcp-server:latest --name ${CLUSTER_NAME}
kind load docker-image finance-mcp-server:latest --name ${CLUSTER_NAME}
kind load docker-image rag-mcp-server:latest --name ${CLUSTER_NAME}
kind load docker-image rag-data-loader:latest --name ${CLUSTER_NAME} # NEW: Load the data loader image

echo "--- Applying Kubernetes manifests ---"

kubectl apply -f ${K8S_DIR}/namespaces.yaml
kubectl apply -f ${K8S_DIR}/secrets.yaml
kubectl apply -f ${K8S_DIR}/configmaps.yaml

# Apply PersistentVolumeClaims before Deployments that use them
mkdir -p ${K8S_DIR}/persistentvolumeclaims # Ensure directory exists for apply
kubectl apply -f ${K8S_DIR}/persistentvolumeclaims/rag-pvc.yaml

# Apply deployments
kubectl apply -f ${K8S_DIR}/deployments/bot-deploy.yaml
kubectl apply -f ${K8S_DIR}/deployments/fastmcp-core-deploy.yaml
kubectl apply -f ${K8S_DIR}/deployments/telegram-mcp-deploy.yaml
kubectl apply -f ${K8S_DIR}/deployments/discord-mcp-deploy.yaml
kubectl apply -f ${K8S_DIR}/deployments/web-mcp-deploy.yaml
kubectl apply -f ${K8S_DIR}/deployments/finance-mcp-deploy.yaml
kubectl apply -f ${K8S_DIR}/deployments/rag-mcp-deploy.yaml

# Apply services
kubectl apply -f ${K8S_DIR}/services/bot-svc.yaml
kubectl apply -f ${K8S_DIR}/services/fastmcp-core-svc.yaml
kubectl apply -f ${K8S_DIR}/services/telegram-mcp-svc.yaml
kubectl apply -f ${K8S_DIR}/services/discord-mcp-svc.yaml
kubectl apply -f ${K8S_DIR}/services/web-mcp-svc.yaml
kubectl apply -f ${K8S_DIR}/services/finance-mcp-svc.yaml
kubectl apply -f ${K8S_DIR}/services/rag-mcp-svc.yaml

# Apply Ingress (uncomment if you have an Ingress controller installed in KinD)
# kubectl apply -f ${K8S_DIR}/ingress/bot-ingress.yaml

echo "--- Deployment to KinD cluster complete! ---"
echo "Verify status with: kubectl get pods -n multi-agent-bot"
echo "And: kubectl get svc -n multi-agent-bot"
echo ""
echo "--- To load initial RAG data, run the Kubernetes Job AFTER all pods are running: ---"
echo "kubectl apply -f ${K8S_DIR}/jobs/rag-data-loader-job.yaml"
echo "Monitor the job's progress with: kubectl get jobs -n multi-agent-bot"
echo "View job logs with: kubectl logs -f job/rag-data-loader-job -n multi-agent-bot"
