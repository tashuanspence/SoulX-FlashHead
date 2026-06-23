import os
from typing import Iterable

try:
    from datadog.dogstatsd import DogStatsd
except ImportError:
    DogStatsd = None

try:
    import boto3
except ImportError:
    boto3 = None

DOGSTATSD_ENABLED = os.environ.get("DD_DOGSTATSD_ENABLED", "true").lower() == "true"
DOGSTATSD_HOST = os.environ.get("DD_AGENT_HOST", "127.0.0.1")
DOGSTATSD_PORT = int(os.environ.get("DD_DOGSTATSD_PORT", 8125))

CLOUDWATCH_METRICS_ENABLED = os.environ.get("CLOUDWATCH_METRICS_ENABLED", "true").lower() == "true"
CLOUDWATCH_NAMESPACE = "SoulX/FlashHead"

_statsd = None
if DOGSTATSD_ENABLED and DogStatsd is not None:
    _statsd = DogStatsd(host=DOGSTATSD_HOST, port=DOGSTATSD_PORT)

_cloudwatch = None
if CLOUDWATCH_METRICS_ENABLED and boto3 is not None:
    _cloudwatch = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _normalize_tags(tags: Iterable[str] | None = None, **tag_fields):
    normalized = list(tags or [])
    for key, value in tag_fields.items():
        if value is None:
            continue
        normalized.append(f"{key}:{value}")
    return normalized


def increment(name: str, value: int = 1, tags: Iterable[str] | None = None, **tag_fields):
    if _statsd is None:
        return
    _statsd.increment(name, value=value, tags=_normalize_tags(tags, **tag_fields))


def gauge(name: str, value: float | int, tags: Iterable[str] | None = None, **tag_fields):
    if _statsd is None:
        return
    _statsd.gauge(name, value, tags=_normalize_tags(tags, **tag_fields))


def distribution(name: str, value: float | int, tags: Iterable[str] | None = None, **tag_fields):
    if _statsd is None:
        return
    _statsd.distribution(name, value, tags=_normalize_tags(tags, **tag_fields))


def generation_metric_tags(generation_type: str, model_type: str, status: str | None = None, endpoint: str | None = None):
    return _normalize_tags(
        generation_type=generation_type,
        model_type=model_type,
        status=status,
        endpoint=endpoint,
        env=os.environ.get("DD_ENV", "unknown"),
        service=os.environ.get("DD_SERVICE", "soulx-flashhead"),
    )


def put_cloudwatch_metric(metric_name: str, value: float, unit: str = "Count", dimensions: dict = None):
    """Emit metric to CloudWatch for ECS auto-scaling."""
    if _cloudwatch is None:
        return
    try:
        metric_data = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
            "Dimensions": [
                {"Name": "ClusterName", "Value": os.environ.get("ECS_CLUSTER_NAME", "unknown")},
                {"Name": "ServiceName", "Value": os.environ.get("ECS_SERVICE_NAME", "unknown")},
            ]
        }
        if dimensions:
            for key, val in dimensions.items():
                metric_data["Dimensions"].append({"Name": key, "Value": val})

        _cloudwatch.put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricData=[metric_data]
        )
    except Exception:
        # Don't fail requests if CloudWatch is down
        pass
