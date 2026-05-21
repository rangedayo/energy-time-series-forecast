"""Global exception handlers."""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.exceptions")


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_input", "detail": str(exc)},
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []))
        errors.append(
            {
                "field": loc,
                "type": err.get("type"),
                "message": err.get("msg"),
            }
        )
    logger.info(
        "Validation error on %s %s: %d field(s)",
        request.method,
        request.url.path,
        len(errors),
    )
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "errors": errors},
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception(
        "Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "예측 처리 중 오류가 발생했습니다.",
        },
    )
