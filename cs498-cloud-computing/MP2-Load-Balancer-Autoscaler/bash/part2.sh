#!/bin/bash

export HOME=/home/ec2-user

sudo yum update

sudo yum install stress-ng -y

sudo yum install htop -y

sudo yum install python3-pip -y

pip3 install flask

sudo yum install git -y

cd /home/ec2-user

git clone https://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@github.com/javi-cardenas/cs498-mp2.git

cd /home/ec2-user/cs498-mp2

python3 serve.py