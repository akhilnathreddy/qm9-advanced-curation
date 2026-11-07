"""The catalogue of validation checks (plan v2 §5).

Severity says what a failure means, not what happens to the record (tiering
is Phase 5's decision):

* ``hard`` — the record is malformed or physically invalid as stored;
* ``identity`` — the stored geometry/properties do not describe the molecule
  the record is labelled as;
* ``soft`` — worth a reviewer's attention; not evidence of an error.

The order of ``CHECKS`` fixes each check's bit in ``flag_bitmask``; append
new checks at the end so existing bits keep their meaning.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    layer: int
    severity: str
    description: str


CHECKS: tuple[Check, ...] = (
    # Layer 1 — format and completeness
    Check("elements_allowed", 1, "hard", "Only H, C, N, O, F atoms"),
    Check("heavy_atom_limit", 1, "hard", "At most 9 heavy atoms"),
    Check("frequency_count", 1, "hard", "Each frequency set has 3N-6 (or 3N-5 if linear) modes"),
    Check("coordinates_finite", 1, "hard", "All coordinates finite"),
    # Layer 2 — chemical identity
    Check("single_fragment", 2, "identity", "Geometry is one connected molecule"),
    Check("gdb_matches_geometry", 2, "identity", "GDB-17 graph equals the graph perceived from 3D"),
    Check("not_uncharacterized", 2, "identity", "Not in the authors' uncharacterized list"),
    Check(
        "relaxed_smiles_matches_geometry", 2, "soft", "Authors' relaxed SMILES matches the 3D graph"
    ),
    # Layer 3 — physics invariants
    Check("gap_consistency", 3, "hard", "gap = LUMO - HOMO"),
    Check("enthalpy_minus_energy", 3, "hard", "H - U = k_B T at 298.15 K"),
    Check("free_energy_order", 3, "hard", "G < H"),
    Check("thermal_order", 3, "hard", "U(298 K) > U(0 K)"),
    Check("mulliken_neutral", 3, "hard", "Mulliken charges sum to zero"),
    Check("zpve_from_frequencies", 3, "hard", "ZPVE = 1/2 sum of a stored frequency set"),
    Check("rotational_constants", 3, "hard", "Rotational constants match the geometry"),
    Check("real_minimum", 3, "hard", "No imaginary frequency in the consistent set"),
    Check(
        "linearity_representation",
        3,
        "soft",
        "A and frequency count agree with geometric linearity",
    ),
    # Layer 4 — statistical and source annotations
    Check(
        "not_statistical_outlier", 4, "soft", "No property with |robust z| > 6 in its size stratum"
    ),
    Check("not_hard_to_converge", 4, "soft", "Not on the authors' difficult-to-converge list"),
)

CHECK_NAMES = tuple(c.name for c in CHECKS)
SEVERITIES = ("hard", "identity", "soft")
