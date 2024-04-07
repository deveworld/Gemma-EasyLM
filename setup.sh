# seq len 2048
export USER='world'
export TPU_NAME='v4-64'
export ZONE='us-central2-b'
export GIT_REPO='https://github.com/deveworld/Gemma-EasyLM'
# export BUCKET='gemago-alpha'
# export DATASET=''

echo "[local] Cloning repository"
gcloud compute tpus tpu-vm ssh $USER@$TPU_NAME \
--zone $ZONE --worker=all --command "git clone ${GIT_REPO}"

echo "[local] Setting up tpu vms"
gcloud compute tpus tpu-vm ssh $USER@$TPU_NAME \
--zone $ZONE --worker=all --command "./home/${USER}/Gemma-EasyLM/scripts/tpu_vm_setup.sh"
