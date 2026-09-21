"""Spaces and occurrences are distinct: labels never establish connectivity."""

from dataclasses import dataclass
from enum import StrEnum

from ricci.core.errors import RicciError


class SpaceKind(StrEnum):
    GEOMETRIC = "geometric"
    COMPONENT = "component"
    BATCH = "batch"


class Variance(StrEnum):
    UP = "contravariant"
    DOWN = "covariant"
    NEUTRAL = "neutral"

    def dual(self) -> "Variance":
        if self is Variance.UP:
            return Variance.DOWN
        if self is Variance.DOWN:
            return Variance.UP
        return self


@dataclass(frozen=True, slots=True)
class IndexSpace:
    """Explicit identity in a declared basis/fiber; equal size is insufficient.

    Reuse an id only for the same space. ``fiber`` may identify a tangent-space
    base point; it is metadata, not a symbolic point or an automatic transport.
    """

    id: str
    size: int
    kind: SpaceKind = SpaceKind.GEOMETRIC
    basis: str = "default"
    fiber: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.basis or type(self.size) is not int or self.size < 1:
            raise RicciError(
                "INVALID_SPACE", "Space requires id, basis and a positive integer size"
            )
        if not isinstance(self.kind, SpaceKind):
            raise RicciError("INVALID_SPACE", "kind must be a SpaceKind")


@dataclass(frozen=True, slots=True)
class Index:
    """One ordered tensor leg. The label is presentation metadata only."""

    label: str
    space: IndexSpace
    variance: Variance = Variance.NEUTRAL

    def __post_init__(self) -> None:
        if not self.label or not isinstance(self.variance, Variance):
            raise RicciError("INVALID_INDEX", "Index requires a label and a Variance")
        geometric = self.space.kind is SpaceKind.GEOMETRIC
        if geometric == (self.variance is Variance.NEUTRAL):
            raise RicciError(
                "INVALID_VARIANCE", "Geometric legs require UP/DOWN; component/batch legs NEUTRAL"
            )

    @property
    def size(self) -> int:
        return self.space.size

    def dual(self) -> "Index":
        """Change the slot type, without changing or conjugating any numerical data."""
        return Index(self.label, self.space, self.variance.dual())
