from pathlib import Path

from qm9_curation import inventory
from qm9_curation.inventory import ArchiveStats

METHANE = (
    "5\n"
    "gdb 1\t157.7118\t157.70997\t157.70699\t0.\t13.21\t-0.3877\t0.1171\t0.5048\t35.3641"
    "\t0.044749\t-40.47893\t-40.476062\t-40.475117\t-40.498597\t6.469\t\n"
    "C\t-0.0126981359\t 1.0858041578\t 0.0080009958\t-0.535689\n"
    "H\t 0.002150416\t-0.0060313176\t 0.0019761204\t 0.133921\n"
    "H\t 1.0117308433\t 1.4637511618\t 0.0002765748\t 0.133922\n"
    "H\t-0.540815069\t 1.4475266138\t-0.8766437152\t 0.133923\n"
    "H\t-0.5238136345\t 1.4379326443\t 0.9063972942\t 0.133923\n"
    "1341.307\t1341.3284\t1341.365\t1562.6731\t1562.7453\t3038.3205\t3151.6034\t3151.6788\t3151.7078\n"
    "C\tC\t\n"
    "InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4\n"
)
NAME = "dsgdb9nsd_000001.xyz"


def test_methane_record_is_clean() -> None:
    stats = ArchiveStats()
    inventory.inspect_record(NAME, METHANE, stats)
    assert stats.header_tags == {"gdb": 1}
    assert stats.freq_class == {"3N-6": 1}
    assert stats.n_heavy == {1: 1}
    assert stats.elements == {"H": 4, "C": 1}
    assert not (stats.bad_line_count or stats.index_mismatch or stats.bad_property_count)
    assert not stats.exponent_records


def test_index_mismatch_and_exponent_are_detected() -> None:
    stats = ArchiveStats()
    text = METHANE.replace("0.0080009958", "8.0*^-3")
    inventory.inspect_record("dsgdb9nsd_000002.xyz", text, stats)
    assert stats.index_mismatch == ["dsgdb9nsd_000002.xyz"]
    assert stats.exponent_records == {"atoms": 1}


def test_doubled_frequencies_and_negative_mode() -> None:
    lines = METHANE.split("\n")
    freqs = lines[7].split("\t")
    lines[7] = "\t".join(["-2.0", *freqs[1:], *freqs])
    stats = ArchiveStats()
    inventory.inspect_record(NAME, "\n".join(lines), stats)
    assert stats.freq_class == {"2x(3N-6) differing halves": 1}
    assert stats.negative_freq == [1]


def test_truncated_record_is_flagged() -> None:
    stats = ArchiveStats()
    inventory.inspect_record(NAME, "\n".join(METHANE.split("\n")[:6]), stats)
    assert stats.bad_line_count == [NAME]


def test_parse_uncharacterized_skips_headers(tmp_path: Path) -> None:
    path = tmp_path / "u.txt"
    path.write_text(
        "List of molecules ...\n====\n#   Index   GDB17 SMILES\n====\n"
        "      58    NC(=N)C#N   NC(=N)C#N   [NH]C(=[NH2])C#N   6.036217\n"
        "      61    NC(=N)C=O   NC(=N)C=O   [NH]C(=[NH2])C=O   5.631463\n====\n",
        encoding="ascii",
    )
    assert inventory.parse_uncharacterized(path) == [58, 61]


def test_parse_atomref(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(
        "====\n  Ele-  ZPVE  U (0 K)\n====\n"
        "   H     0.000000     -0.500273      -0.498857      -0.497912      -0.510927     2.981\n",
        encoding="ascii",
    )
    assert inventory.parse_atomref(path)["H"]["u0"] == -0.500273
