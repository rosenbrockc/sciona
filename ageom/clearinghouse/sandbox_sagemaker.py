"""Heavy/GPU-tier sandbox executor using SageMaker Processing Jobs."""

from __future__ import annotations

import logging

from ageom.clearinghouse.models import SandboxPayload, SandboxResult
from ageom.clearinghouse.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

# Tier-specific instance types
TIER_INSTANCES: dict[str, str] = {
    "heavy": "ml.m5.4xlarge",
    "gpu": "ml.g5.xlarge",
}

# Tier-specific max runtime in seconds
TIER_MAX_RUNTIME: dict[str, int] = {
    "heavy": 7200,   # 2 hours
    "gpu": 14400,    # 4 hours
}


class SageMakerSandboxExecutor:
    """Execute CDGs via SageMaker Processing Jobs (Heavy/GPU tiers).

    Heavy tier: ml.m5.4xlarge (16 vCPU, 64 GB)
    GPU tier: ml.g5.xlarge (1x A10G, 24 GB VRAM)
    """

    def __init__(
        self,
        *,
        tier: str = "heavy",
        role_arn: str = "",
        s3_bucket: str = "ageom-platform",
        image_uri: str = "",
        region: str = "us-east-1",
    ) -> None:
        self._tier = tier
        self._role_arn = role_arn
        self._s3_bucket = s3_bucket
        self._image_uri = image_uri
        self._region = region

    async def execute(self, payload: SandboxPayload) -> SandboxResult:
        """Start a SageMaker Processing Job.

        Jobs are async — this method starts the job and polls for completion.
        For production, use a callback/webhook pattern instead.
        """
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            return SandboxResult(error="boto3 not installed")

        try:
            sm_client = boto3.client("sagemaker", region_name=self._region)
            s3_client = boto3.client("s3", region_name=self._region)

            job_name = f"ageom-{self._tier}-{payload.submission_id[:8]}"
            s3_key = f"sandbox_payloads/{payload.bounty_id}/{payload.submission_id}.json"

            s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=s3_key,
                Body=payload.model_dump_json().encode(),
            )

            instance_type = TIER_INSTANCES.get(self._tier, "ml.m5.4xlarge")
            max_runtime = TIER_MAX_RUNTIME.get(self._tier, 7200)

            sm_client.create_processing_job(
                ProcessingJobName=job_name,
                ProcessingResources={
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": instance_type,
                        "VolumeSizeInGB": 50,
                    }
                },
                AppSpecification={
                    "ImageUri": self._image_uri,
                    "ContainerArguments": [
                        "--s3-bucket", self._s3_bucket,
                        "--s3-key", s3_key,
                    ],
                },
                RoleArn=self._role_arn,
                NetworkConfig={"EnableNetworkIsolation": True},
                StoppingCondition={"MaxRuntimeInSeconds": max_runtime},
            )

            return SandboxResult(
                error="",
                trace={"sagemaker_job_name": job_name, "status": "started"},
            )
        except Exception as exc:
            logger.exception("SageMaker sandbox execution failed")
            return SandboxResult(error=str(exc))
