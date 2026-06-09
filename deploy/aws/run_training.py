import boto3
import os

# AWS_REGION        = "ap-south-1"
# ECR_REGION        = "ap-south-1"
AWS_REGION        = "us-east-1"
ECR_REGION        = "us-east-1"
INSTANCE_TYPE     = "ml.g4dn.xlarge"
# INSTANCE_TYPE     = "ml.m5.xlarge"
TRAINING_JOB_NAME = "neuroscan-train-v12"
AWS_ACCOUNT       = boto3.client("sts").get_caller_identity()["Account"]
IMAGE_URI         = f"{AWS_ACCOUNT}.dkr.ecr.{ECR_REGION}.amazonaws.com/neuroscan:latest"
ROLE_ARN          = f"arn:aws:iam::{AWS_ACCOUNT}:role/SageMakerNeuroScanRole"
# DATA_BUCKET       = f"neuroscan-data-{AWS_ACCOUNT}"
# OUT_BUCKET        = f"neuroscan-artifacts-{AWS_ACCOUNT}"
DATA_BUCKET       = f"neuroscan-data-{AWS_ACCOUNT}-use1"
OUT_BUCKET        = f"neuroscan-artifacts-{AWS_ACCOUNT}-use1"

sm = boto3.client("sagemaker", region_name=AWS_REGION)

sm.create_training_job(
    TrainingJobName=TRAINING_JOB_NAME,
    AlgorithmSpecification={
        "TrainingImage":       IMAGE_URI,
        "TrainingInputMode":   "File",
        "ContainerEntrypoint": ["python", "train_swin.py"],
    },
    RoleArn=ROLE_ARN,
    InputDataConfig=[{
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType":             "S3Prefix",
                "S3Uri":                  f"s3://{DATA_BUCKET}/adni/",
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }],
    OutputDataConfig={
        "S3OutputPath": f"s3://{OUT_BUCKET}/"
    },
    ResourceConfig={
        "InstanceType":   INSTANCE_TYPE,
        "InstanceCount":  1,
        "VolumeSizeInGB": 50,
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 86400 # 24 hours
    },
    Environment={
        "S3_ARTIFACTS_BUCKET": OUT_BUCKET,
        "SM_MODEL_DIR":        "/opt/ml/model",
        "SM_CHANNEL_TRAIN":    "/opt/ml/input/data/train",
        "OVERRIDE_EPOCHS":     "5",   # just to verify pipeline works
    },
)

print(f"Training job submitted.")
print(f"Monitor -> https://{ECR_REGION}.console.aws.amazon.com/sagemaker/home?region={ECR_REGION}#/jobs/{TRAINING_JOB_NAME}")
print(f"Logs -> aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {TRAINING_JOB_NAME} --follow --region {ECR_REGION}")