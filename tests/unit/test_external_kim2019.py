import io
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from qm9_curation.config import load_config
from qm9_curation.external import kim2019

MEMBERS = load_config("external")["kim2019"]["members"]
# Published G4(MP2) 0 K atom energies (Hartree) and the multiplicities the authors used.
ATOMS = {
    "H": (-0.5020936, -0.5002728, 2),
    "C": (-37.7942028, -37.8467711, 3),
    "N": (-54.439864, -54.4795287, 2),
    "O": (-75.0024828, -75.0645779, 3),
    "F": (-99.6596857, -99.7187294, 2),
}


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_archive(
    path: Path,
    molecules: Sequence[tuple[int, str, float, float, float]],
    atoms: Mapping[str, tuple[float, float, int]] = ATOMS,
    not_converged: Sequence[int] = (),
    multi_molecule: Sequence[int] = (),
    duplicate_pairs: Sequence[tuple[int, int]] = (),
) -> Path:
    """Archive laid out like the authors' (rows: qm9_id, Chem, G4MP2, B3LYP, ZPE)."""
    head = ',index,Chem,"B3LYP/6-31g(2df,p)",G4MP2,Zero-point correction\n'
    mol_csv = head + "".join(f"0,{i},{c},{b},{g},{z}\n" for i, c, g, b, z in molecules)
    atom_csv = ',index,Chem,"B3LYP/6-31g(2df,p)",G4MP2\n' + "".join(
        f"0,0,{e},{b},{g}\n" for e, (g, b, _) in atoms.items()
    )
    refs = {e: g for e, (g, _, _) in atoms.items()}
    authors = ",index,Chem,G4MP2\n" + "".join(
        f"0,{i},{c},{sum(n * refs[e] for e, n in kim2019.formula_counts(c).items()) - g}\n"
        for i, c, g, _, _ in molecules
    )
    files = {
        MEMBERS["molecules"]: mol_csv,
        MEMBERS["atoms"]: atom_csv,
        MEMBERS["authors_atomization"]: authors,
        # numpy.savetxt format, as in the real lists
        MEMBERS["not_converged"]: "".join(f"{float(i):.18e}\n" for i in not_converged),
        MEMBERS["multi_molecule"]: "".join(f"{i}\n" for i in multi_molecule),
        MEMBERS["duplicate_pairs"]: "".join(f"{a} {b}\n" for a, b in duplicate_pairs),
    }
    for e, (_, _, mult) in atoms.items():
        files[MEMBERS["atom_logs"].format(element=e)] = (
            f" Symbolic Z-matrix:\n Charge =  0 Multiplicity = {mult}\n {e} 0. 0. 0.\n"
        )
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            _add(tar, name, text.encode("ascii"))
        _add(tar, "data_publication/parse.py", b"raise SystemExit('never read')\n")
    return path


def test_parse_reads_tables_lists_and_multiplicities(tmp_path: Path) -> None:
    archive = write_archive(
        tmp_path / "a.tar.gz",
        [(1, "C1H4", -40.4276622, -40.5236807, 0.04475), (2, "H3N1", -56.47, -56.56, 0.0343)],
        not_converged=[212, 7],
        multi_molecule=[5],
        duplicate_pairs=[(933, 937)],
    )
    kim = kim2019.load(archive, MEMBERS)
    assert kim.molecules["qm9_id"].to_list() == [1, 2]
    assert kim.molecules.row(0, named=True)["g4mp2_u0_ha"] == -40.4276622
    assert kim.not_converged == [7, 212]
    assert kim.multi_molecule == [5]
    assert kim.duplicate_pairs == [(933, 937)]
    mult = dict(kim.atoms.select("element", "multiplicity").iter_rows())
    assert mult == {"H": 2, "C": 3, "N": 2, "O": 3, "F": 2}


def test_only_configured_members_are_read(tmp_path: Path) -> None:
    archive = write_archive(tmp_path / "a.tar.gz", [(1, "C1H4", -40.4, -40.5, 0.04)])
    raw = kim2019.read_members(archive, MEMBERS)
    assert "data_publication/parse.py" not in raw
    assert {k for k in raw if k.startswith("atom_logs:")} == {
        f"atom_logs:{e}" for e in kim2019.ELEMENTS
    }


def test_directory_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("data_publication/result.csv")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    with pytest.raises(ValueError, match="not a regular file"):
        kim2019.read_members(archive, {"molecules": "data_publication/result.csv"})


def test_formula_counts() -> None:
    assert kim2019.formula_counts("C1H4") == {"C": 1, "H": 4}
    assert kim2019.formula_counts("C7H10N1O1F1") == {"C": 7, "H": 10, "N": 1, "O": 1, "F": 1}
    with pytest.raises(ValueError):
        kim2019.formula_counts("CH4")  # counts are always explicit in the source


def test_multiplicity_requires_a_line() -> None:
    assert kim2019.multiplicity(b" Charge =  0 Multiplicity = 4\n") == 4
    with pytest.raises(ValueError):
        kim2019.multiplicity(b"no such line")
