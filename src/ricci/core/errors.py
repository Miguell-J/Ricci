"""Stable domain errors, independent of MCP and tensor backends."""


class RicciError(ValueError):
    """A rejected scientific operation with a machine-readable code and location."""

    def __init__(self, code: str, message: str, *, path: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def descriptor(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self), "path": self.path}
