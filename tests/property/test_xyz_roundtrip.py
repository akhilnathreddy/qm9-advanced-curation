from hypothesis import given
from hypothesis import strategies as st

from qm9_curation.parse.xyz import Atom, Record, parse_record, serialize_record


def number() -> st.SearchStrategy[str]:
    mantissa = st.from_regex(r"-?\d{1,4}\.\d{0,10}", fullmatch=True)
    exponent = st.from_regex(r"-?\d{1,4}\.\d{1,6}\*\^-\d", fullmatch=True)
    return st.one_of(mantissa, exponent)


@st.composite
def records(draw: st.DrawFn) -> Record:
    n_atoms = draw(st.integers(min_value=3, max_value=12))
    linear = draw(st.booleans())
    n_modes = 3 * n_atoms - (5 if linear else 6)
    n_sets = draw(st.sampled_from([1, 2]))
    atoms = tuple(
        Atom(draw(st.sampled_from("HCNOF")), *(draw(number()) for _ in range(4)))
        for _ in range(n_atoms)
    )
    word = st.text(alphabet="CNOF()=#[]@+-123cno", min_size=1, max_size=20)
    return Record(
        index=draw(st.integers(min_value=1, max_value=133885)),
        properties=tuple(draw(number()) for _ in range(15)),
        atoms=atoms,
        frequencies=tuple(draw(number()) for _ in range(n_sets * n_modes)),
        smiles_gdb=draw(word),
        smiles_relaxed=draw(word),
        inchi_gdb="InChI=1S/" + draw(word),
        inchi_relaxed="InChI=1S/" + draw(word),
    )


@given(records())
def test_serialize_then_parse_is_identity(record: Record) -> None:
    text = serialize_record(record)
    assert parse_record(text, "generated") == record
    assert serialize_record(parse_record(text, "generated")) == text
