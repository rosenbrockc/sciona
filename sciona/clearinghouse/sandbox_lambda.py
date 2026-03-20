"""Standard-tier sandbox executor using AWS Lambda."""

from __future__ import annotations

import json
import logging

from sciona.clearinghouse.models import SandboxPayload, SandboxResult
from sciona.clearinghouse.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)


class LambdaSandboxExecutor:
    """Execute CDGs via AWS Lambda (Standard tier).

    Lambda provides: 15 min timeout, 4-10 GB memory, VPC with no
    internet egress, deterministic Python environment.
    """

    def __init__(
        self,
        *,
        function_name: str = "sciona-sandbox-standard",
        s3_bucket: str = "sciona-platform",
        region: str = "us-east-1",
    ) -> None:
        self._function_name = function_name
        self._s3_bucket = s3_bucket
        self._region = region

    async def execute(self, payload: SandboxPayload) -> SandboxResult:
        """Invoke Lambda synchronously with the CDG payload.

        Flow:
        1. Upload payload to S3
        2. Invoke Lambda with S3 reference
        3. Lambda: download payload + blind split, execute CDG, upload results
        4. Read results from S3
        """
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            return SandboxResult(error="boto3 not installed")

        try:
            lambda_client = boto3.client("lambda", region_name=self._region)
            s3_key = f"sandbox_payloads/{payload.bounty_id}/{payload.submission_id}.json"

            s3_client = boto3.client("s3", region_name=self._region)
            s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=s3_key,
                Body=payload.model_dump_json().encode(),
            )

            response = lambda_client.invoke(
                FunctionName=self._function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps({"s3_bucket": self._s3_bucket, "s3_key": s3_key}),
            )

            result_payload = json.loads(response["Payload"].read())
            if "errorMessage" in result_payload:
                return SandboxResult(error=result_payload["errorMessage"])

            return SandboxResult.model_validate(result_payload)
        except Exception as exc:
            logger.exception("Lambda sandbox execution failed")
            return SandboxResult(error=str(exc))
