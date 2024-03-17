#! /usr/bin/env bash

#sh <(curl -L https://nixos.org/nix/install) --daemon 
sudo tee -a /etc/nix/nix.conf > /dev/null <<EOF 
trusted-users = root $(whoami) 
extra-experimental-features = nix-command flakes
EOF
sudo systemctl restart nix-daemon
