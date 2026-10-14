aws ec2 run-instances \
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
    }'