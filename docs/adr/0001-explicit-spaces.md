# ADR 0001: Explicit spaces and port identity

Status: accepted for the alpha foundation.

Equal dimensions and labels cannot distinguish tangent fibers, coordinate bases,
material/spatial spaces or internal features. The core therefore identifies a space
by explicit id and consistent size/kind/basis/fiber metadata. Each leg has variance;
an edge connects port occurrences and has no single variance of its own.

Consequences: relabeling dummy indices cannot change connectivity; invalid duality
fails before execution. Users must explicitly declare compatible spaces. A name-based
Einstein convenience frontend may be added later, lowering to the same explicit IR.
