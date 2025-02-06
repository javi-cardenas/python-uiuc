#!/bin/bash

### General Shell Script Rules
# 1. shebang (#!) identifies this is a shell script, 
#    more portable -> (#!/bin/sh), more advanced -> (#!/bin/bash)
# 2. make the file executable 
#   2.1 cd into the directory that contains the file you want to make executable
#   2.1 run this command: chmod +x YourScriptName.sh
# 3 run the script by entering its pathname
#   3.1 Ex. ~/Documents/Dev/YourScriptName.sh or ./YourScriptName.sh

### Secure EC2 SSH Script
# This script connects to an AWS EC2 instance securely.
# Ensure you have set the correct environment variables before running.

# 1. place key pair .pem file in the same directory as this script
#   1.1 When you first use a .pem file, you will get a bad permissons error because
#       the file is unprotected. Run this command to solve the bad permissions error:
#       chmod 400 YourEC2KeyPair.pem
# 2. update your environment variables
# 3. run the script using ./YourScriptName.sh
KEY_PAIR="myec2key.pem"

if [[ -z "$KEY_PAIR" ]]; then
    echo "ERROR: KEY_PAIR environment variable is not set."
    exit 1
fi

# Ensure EC2_PUBLIC_IP4 is set
if [[ -z "$EC2_PUBLIC_IP4" ]]; then
    echo "ERROR: EC2_PUBLIC_IP4 environment variable is not set."
    exit 1
fi

# Confirm before connecting
read -p "Connect to EC2 at $EC2_PUBLIC_IP4 using $KEY_PAIR? (yes/no) " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborting."
    exit 1
fi

# Connect to EC2 instance
ssh -i "$KEY_PAIR" -o "StrictHostKeyChecking=accept-new" ec2-user@"$EC2_PUBLIC_IP4"
# StrictHostKeyChecking=accept-new
# TL;DR
# Prevents automatic connections to potentially compromised instances.
# Once the key is stored in known_hosts, SSH behaves normally and won’t prompt.

# Behavior:
# If the EC2 instance is never seen before, it prompts once and automatically adds it
# to ~/.ssh/known_hosts. If the host key changes in the future, SSH will warn you and
# refuse to connect (preventing MITM attacks), i.e. more secure than StrictHostKeyChecking=no.