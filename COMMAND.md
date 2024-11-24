## Enter Overlay and adjust environment
chmod +x /home/yw4142/starter_scripts/sing_rw.sh
/home/yw4142/starter_scripts/sing_rw.sh
mamba env create -f conda_env.yaml

## Exit and enter read-only overlay
chmod +x /home/yw4142/starter_scripts/sing_ro.sh
/home/yw4142/starter_scripts/sing_ro.sh
mamba activate dl_final

## Wandb Login Error
curl -o ~/cacert.pem https://curl.se/ca/cacert.pem
export SSL_CERT_FILE=/home/$USER/cacert.pem ## put this inside .bashrc
source ~/.bashrc