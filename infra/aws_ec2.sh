set -euo pipefail

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "ami-0d80e0723483d86f6" \
    --instance-type "g5.xlarge" \
    --key-name "aws-personal-eus1" \
    --ebs-optimized \
    --count 1 \
    --block-device-mappings '[
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "Encrypted": false,
                "DeleteOnTermination": true,
                "Iops": 3000,
                "SnapshotId": "snap-0ffb61caba19be95c",
                "VolumeSize": 100,
                "VolumeType": "gp3",
                "Throughput": 125
            }
        }
    ]' \
    --network-interfaces '[
        {
            "AssociatePublicIpAddress": true,
            "DeviceIndex": 0,
            "Groups": ["sg-04ba2b31492f9fb9e"]
        }
    ]' \
    --tag-specifications '[
        {
            "ResourceType": "instance",
            "Tags": [
                {
                    "Key": "Name",
                    "Value": "ubuntudlserver"
                }
            ]
        }
    ]' \
    --metadata-options '{
        "HttpEndpoint": "enabled",
        "HttpPutResponseHopLimit": 2,
        "HttpTokens": "required"
    }' \
    --private-dns-name-options '{
        "HostnameType": "ip-name",
        "EnableResourceNameDnsARecord": true,
        "EnableResourceNameDnsAAAARecord": false
    }' \
    --user-data '#!/bin/bash
sudo -u ubuntu -i -- bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Launched instance $INSTANCE_ID, waiting for it to enter 'running' state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance $INSTANCE_ID is running. Public IP: $PUBLIC_IP"
echo "Note: --user-data (installing uv) runs in the background after boot;"
echo "give it a few seconds after SSH'ing in before 'source \$HOME/.local/bin/env'."