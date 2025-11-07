#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <HUGGINGFACE_TOKEN>"
    exit 1
fi

HF_TOKEN="$1"

VM_USER="student-admin"
VM_HOST="paffenroth-23.dyn.wpi.edu"
VM_PORT=22016
DEPLOY_SCRIPT_URL="https://raw.githubusercontent.com/keerthi042000/CaseStudy2/main/deploy.sh"
CUSTOM_KEY="$HOME/.ssh/CaseStudy2_16"
DEFAULT_KEY="$HOME/.ssh/student-admin_key"
LOG_FILE="$HOME/MLOPS/CaseStudy2/cron_recovery.log"

# List of public keys to allow access (add your teammates' public keys)
PUBLIC_KEYS=(
"$HOME/.ssh/CaseStudy2_16.pub"
"$HOME/.ssh/CaseStudy2_16_Keerthi.pub"
"$HOME/.ssh/CaseStudy2_16_GG.pub"
"$HOME/.ssh/CaseStudy2_16_HP.pub"
)

echo "$(date): Starting cron recovery..." >> "$LOG_FILE"

# --- Try connecting with custom key ---
if ssh -i "$CUSTOM_KEY" -p $VM_PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "echo 1" &>/dev/null; then
    echo "$(date): VM reachable with custom key. Checking app..." >> "$LOG_FILE"

    ssh -i "$CUSTOM_KEY" -p $VM_PORT $VM_USER@$VM_HOST "
        if lsof -i:7860 > /dev/null; then
            echo 'App running on port 7860. No action.'
            exit 0
        else
            echo 'App not running. Redeploying...'
            curl -s -L $DEPLOY_SCRIPT_URL -o ~/projects/deploy.sh
            bash ~/projects/deploy.sh $HF_TOKEN
        fi
    "

else
    echo "$(date): Custom key failed. Trying default key..." >> "$LOG_FILE"

    if ssh -i "$DEFAULT_KEY" -p $VM_PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "echo 1" &>/dev/null; then
        echo "$(date): Connected with default key. Updating authorized_keys..." >> "$LOG_FILE"

        # ssh -i "$DEFAULT_KEY" -p $VM_PORT $VM_USER@$VM_HOST '
        #     mkdir -p ~/.ssh
        #     chmod 700 ~/.ssh
        # '

        # for pubkey in "${PUBLIC_KEYS[@]}"; do
        #     ssh -i "$DEFAULT_KEY" -p $VM_PORT $VM_USER@$VM_HOST "cat >> ~/.ssh/authorized_keys" < "$pubkey"
        # done

        # Step 1: Create temp file

        ssh -i "$DEFAULT_KEY" -p $VM_PORT $VM_USER@$VM_HOST "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat > ~/.ssh/authorized_keys <<'EOF'
$(cat "$HOME/.ssh/CaseStudy2_16.pub")
$(cat "$HOME/.ssh/CaseStudy2_16_Keerthi.pub")
$(cat "$HOME/.ssh/CaseStudy2_16_GG.pub")
$(cat "$HOME/.ssh/CaseStudy2_16_HP.pub")"


        ssh -i "$CUSTOM_KEY" -p $VM_PORT $VM_USER@$VM_HOST 'chmod 600 ~/.ssh/authorized_keys'

        # ssh -i "$DEFAULT_KEY" -p $VM_PORT $VM_USER@$VM_HOST 'chmod 600 ~/.ssh/authorized_keys'

        echo "$(date): Redeploying app..." >> "$LOG_FILE"
        ssh -i "$CUSTOM_KEY" -p $VM_PORT $VM_USER@$VM_HOST "
            curl -s -L $DEPLOY_SCRIPT_URL -o ~/deploy.sh
            bash ~/deploy.sh $HF_TOKEN
        "
    else
        echo "$(date): VM unreachable with both keys." >> "$LOG_FILE"
    fi
fi
