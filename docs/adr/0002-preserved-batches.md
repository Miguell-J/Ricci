# ADR 0002: Retained batches are not contraction edges

Status: accepted for the alpha foundation.

Shared indices in attention include both contracted features and retained batches.
Matching labels cannot determine which role is intended. BatchGroup explicitly
aligns batch-space ports, with exactly one declared output representative. Ordinary
edges sum compatible component or dual geometric legs.

Consequences: no accidental summation of samples/heads; multiple operands may share
a retained axis. Size-one broadcasting, batch reductions and within-node diagonals
remain unsupported until their semantics and tests are specified.
