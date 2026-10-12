"""Legal-form extraction for French names.

French legal forms name DIFFERENT legal entities (a SARL and an SA are not the
same company), whereas the view builder treats them as droppable suffixes and
dotted spellings ("s.a.r.l.") shred into single letters. This recovers the
canonical form so a pair can be tested for a form CONFLICT (both present,
different), which is kept apart from a form being dropped (one side missing).
"""
import re

FORMS = {"sarl", "sas", "sasu", "sa", "eurl", "ei", "eirl", "sci", "snc", "scop",
         "selarl", "selas", "sca", "scs", "gie", "scp", "sem"}
_DOTTED = re.compile(r"\b[a-z0-9](?:\s?\.\s?[a-z0-9]\b)+\.?")   # s.a.s / s.a.r.l. / e. i.
_SPLIT = re.compile(r"[^0-9a-zà-ÿ]+")
_HOMO = str.maketrans({"5": "s", "0": "o", "1": "l", "3": "e", "4": "a"})


def legal_form(name: str) -> str:
    """Canonical legal form token(s) found in a name, '|'-joined; '' if none."""
    if not name:
        return ""
    s = name.lower()
    s = _DOTTED.sub(lambda m: " " + re.sub(r"[\s.]", "", m.group(0)) + " ", s)
    out = []
    for tok in _SPLIT.split(s):
        if not tok:
            continue
        t = tok if tok in FORMS else tok.translate(_HOMO)
        if t in FORMS and t not in out:
            out.append(t)
    return "|".join(sorted(out))


def relation(a: str, b: str) -> int:
    """0 = both missing, 1 = one side missing, 2 = equal, 3 = conflict."""
    if not a and not b:
        return 0
    if not a or not b:
        return 1
    return 2 if set(a.split("|")) & set(b.split("|")) else 3


# India/US forms: synonyms collapse to one canonical form; used to measure the
# generator's legal-form behaviour on labelled data.
_IU = {"llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
       "pc": "pc", "pllc": "pllc", "lp": "lp", "llp": "llp", "ltd": "ltd", "limited": "ltd",
       "pvt": "pvt", "private": "pvt", "pa": "pa", "plc": "plc", "opc": "opc"}


def legal_form_iu(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = _DOTTED.sub(lambda m: " " + re.sub(r"[\s.]", "", m.group(0)) + " ", s)
    out = set()
    for tok in _SPLIT.split(s):
        f = _IU.get(tok)
        if f:
            out.add(f)
    if "pvt" in out:                      # "pvt ltd" / "private limited" is one form
        out.discard("ltd")
    return "|".join(sorted(out))
