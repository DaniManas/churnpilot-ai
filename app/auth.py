from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

# Tells FastAPI to look for the API key in the request header called "X-API-Key"
# auto_error=False means we handle the error ourselves (better error messages)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    FastAPI dependency for API key authentication.

    Usage in a router:
        @router.post("/predict")
        def predict(api_key: str = Depends(verify_api_key)):
            ...

    If the key is missing or invalid, this raises HTTP 401 before the
    endpoint function even runs — the request never reaches your logic.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Add 'X-API-Key: <your-key>' to your request headers."
        )

    if api_key not in settings.get_valid_keys():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )

    return api_key
