"""Multi-view normalization for business names and addresses.

Design rule: every transformation is an ADDITIONAL view. Raw fields are never
overwritten, because each transform destroys evidence some other view needs.

The noise operators observed in the training pairs that these views target:
  names    - case, legal-suffix add/drop, punctuation/bracket injection,
             accent injection (Sanderson -> Sanderson), homoglyph typos
             (5anderson, Fa1cone, 8rigitte), token duplication (Tri-State
             Tri-State), token transposition, truncation, '#handle' and
             'domain.com' forms, 'M/s'/'The'/'f/k/a' prefixes, acronyms,
             generic-token swaps (Hospital -> Center), Indic transliteration.
  address  - component reordering, abbreviation (Rd/Road, Ave/Avenue),
             state alias (IA/Iowa, DL/Delhi/dilli), ordinal<->word
             (7th/Seventh), leading-zero padding (0033), leading-digit drop
             (513 -> 13), ranges (1802-1804), fractions (7494 1/2),
             'NULL'/'<NULL>' literals, added components (PMB, PO BOX, Unit).
"""
from __future__ import annotations
import re
import unicodedata

# --------------------------------------------------------------- primitives
_WS = re.compile(r"\s+")
_COMBINING = dict.fromkeys(
    i for i in range(0x110000) if unicodedata.combining(chr(i))
)
# characters commonly injected as separators / decoration
def _mark_class() -> str:
    """Regex class body covering Unicode combining marks (categories M*).

    Python's ``re`` has no \\p{M}, and ``\\w`` excludes marks: Devanagari
    matras are category Mc/Mn, so a naive ``[^\\w\\s]`` strips them and
    shreds every Indic word into single consonants. Marks are therefore added
    back explicitly. Scanning to 0x20000 covers every script in this data.
    """
    ranges, start, prev = [], None, None
    for cp in range(0x0300, 0x20000):
        if unicodedata.category(chr(cp))[0] == "M":
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            ranges.append((start, prev)); start = None
    if start is not None:
        ranges.append((start, prev))
    return "".join(f"\\U{a:08x}-\\U{b:08x}" for a, b in ranges)


_MARKS = _mark_class()
_PUNCT_TO_SPACE = re.compile(rf"[^\w\s{_MARKS}]", flags=re.UNICODE)
_NONALNUM = re.compile(rf"[^\w{_MARKS}]|_", flags=re.UNICODE)
_DIGITS = re.compile(r"\d+")

# homoglyph typos seen in the data (digit written for a letter)
_HOMOGLYPH = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a",
                            "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"})

# literal null markers that appear as address components
_NULL_TOKENS = {"null", "nul", "na", "n/a", "none", "nil", "<null>", "(null)"}


def nfkc_fold(s: str) -> str:
    """Unicode NFKC + casefold + whitespace collapse. The base of every view."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    return _WS.sub(" ", s).strip()


def strip_accents(s: str) -> str:
    """Drop combining marks that decorate LATIN bases only.

    Accent injection (Sanderson -> Sanderson) is a noise operator, so folding
    helps. Indic matras are NOT decoration - dropping them turns
    'dilli' into nonsense - so combining marks are kept for non-Latin bases.
    """
    if not s:
        return ""
    if s.isascii():
        return s
    out = []
    latin_base = False
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch):
            if not latin_base:
                out.append(ch)
            continue
        latin_base = ("a" <= ch <= "z") or ("A" <= ch <= "Z")
        out.append(ch)
    return "".join(out)


def basic(s: str) -> str:
    """Punctuation -> space, accents folded, whitespace collapsed."""
    if not s:
        return ""
    return _WS.sub(" ", _PUNCT_TO_SPACE.sub(" ", strip_accents(s))).strip()


def compact(s: str) -> str:
    """Letters+digits only. Recovers '#breannesanderson' and 'foo-bar.com'."""
    if not s:
        return ""
    return _NONALNUM.sub("", strip_accents(s))


def homoglyph_fold(s: str) -> str:
    """Map digits that stand in for letters. Recovers 5anderson/Fa1cone/8rigitte."""
    return s.translate(_HOMOGLYPH)


def tokens(s: str) -> list[str]:
    return basic(s).split()


def dedupe_tokens(toks: list[str]) -> list[str]:
    """Collapse adjacent and global duplicates ('tri-state tri-state')."""
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def token_sorted(toks: list[str]) -> str:
    """Word-order invariant view (component reordering is pervasive here)."""
    return " ".join(sorted(toks))


def acronym(toks: list[str]) -> str:
    """First letters of content tokens: 'Dream Systems' -> 'ds'."""
    return "".join(t[0] for t in toks if t)


# --------------------------------------------------------------- name views
# 'also known as' markers: the real name follows the marker
_FKA = re.compile(r"\b(?:f\s*/?\s*k\s*/?\s*a|d\s*/?\s*b\s*/?\s*a|a\s*/?\s*k\s*/?\s*a)\b")
_LEAD_JUNK = re.compile(r"^(?:#|m\s*/\s*s\.?\s+|the\s+|mr\.?\s+|mrs\.?\s+|ms\.?\s+|dr\.?\s+)+")
_DOMAIN_TAIL = re.compile(r"\.(?:com|net|org|co|in|io|biz|info|fr)$")


def name_pre(raw: str) -> str:
    """NFKC-folded name with handle/honorific/domain decoration removed.

    'f/k/a' splits are kept as the RIGHT side, because the training pairs show
    'Noviecto F/K/A Zenia G. Robertson' matching S1 'Zenia G. Robertson'.
    """
    s = nfkc_fold(raw)
    if not s:
        return ""
    m = _FKA.search(s)
    if m:
        s = s[m.end():].strip() or s
    s = _LEAD_JUNK.sub("", s).strip()
    s = _DOMAIN_TAIL.sub("", s)
    return s


def name_views(raw: str, suffixes: frozenset[str], generics: frozenset[str]) -> dict:
    """All parallel name representations used by retrieval and features."""
    pre = name_pre(raw)
    bas = basic(pre)
    toks = bas.split()
    ded = dedupe_tokens(toks)
    # legal suffixes are dropped only when something survives
    core = [t for t in ded if t not in suffixes]
    if not core:
        core = ded
    # generic business words ('center', 'services') are swapped by the noise model
    content = [t for t in core if t not in generics]
    if not content:
        content = core
    cpt = compact(pre)
    return {
        "pre": pre,
        "basic": bas,
        "compact": cpt,
        "compact_hg": homoglyph_fold(cpt),
        "toks": ded,
        "core": core,
        "content": content,
        "sorted": token_sorted(ded),
        "core_sorted": token_sorted(core),
        "content_sorted": token_sorted(content),
        "acronym": acronym(ded),
        "n_tok": len(toks),
    }


# ------------------------------------------------------------ address views
_ORD_SUFFIX = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_RANGE = re.compile(r"^(\d+)[-/](\d+)$")


def _num_variants(tok: str) -> list[str]:
    """Canonical numeric forms of a token ('0033'->33, '1802-1804'->1802,1804)."""
    out = []
    for m in _DIGITS.finditer(tok):
        d = m.group(0).lstrip("0") or "0"
        out.append(d)
    return out


def addr_views(raw: str, alias: dict[str, str], generics: frozenset[str]) -> dict:
    """All parallel address representations.

    `alias` is the data-mined token equivalence map (rd->road, ia->iowa,
    dilli->delhi, eighth->8); it is learned from ground-truth pairs, never
    hard-coded, so it extends to France automatically.
    """
    pre = nfkc_fold(raw)
    bas = basic(pre)
    toks = [t for t in bas.split() if t not in _NULL_TOKENS]
    # canonicalise ordinals to bare digits, then apply the mined alias map
    canon = []
    for t in toks:
        m = _ORD_SUFFIX.match(t)
        if m:
            t = m.group(1)
        t = alias.get(t, t)
        canon.append(t)
    canon = [(t.lstrip("0") or "0") if t.isdigit() else t for t in canon]
    nums = []
    for t in canon:
        nums.extend(_num_variants(t))
    words = [t for t in canon if not t.isdigit()]
    content = [t for t in words if t not in generics]
    return {
        "pre": pre,
        "basic": bas,
        "canon": " ".join(canon),
        "compact": compact(pre),
        "toks": canon,
        "words": words,
        "content": content,
        "sorted": token_sorted(canon),
        "words_sorted": token_sorted(words),
        "nums": nums,
        "numset": sorted(set(nums)),
        "first_num": nums[0] if nums else "",
        "missing": 1 if not toks else 0,
        "n_tok": len(toks),
    }
