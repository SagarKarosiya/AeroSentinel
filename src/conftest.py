import pytest
from inference_engine import AeroSentinelInference


@pytest.fixture(scope="session")
def engine():
    return AeroSentinelInference()


@pytest.fixture(params=["holybro", "speedybee"])
def hardware(request):
    return request.param
