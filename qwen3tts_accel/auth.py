from __future__ import annotations


def validate_bearer_token(authorization: str | None, expected_api_key: str | None) -> None:
    if not expected_api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise PermissionError("Missing bearer token.")

    token = authorization[len("Bearer ") :].strip()
    if token != expected_api_key:
        raise PermissionError("Invalid bearer token.")
