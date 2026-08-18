#!/usr/bin/env python3
"""One definition of which taxonomic group belongs to which domain of life.

This file exists because getting it wrong twice cost two wrong conclusions in one
session. Both were the same mistake: hard-coding "the 8 bacterial targets of the
gammaproteobacteria ladder" as if it were the set of bacterial groups.

  * It omits gammaproteobacteria, which is not a target of its own ladder but IS
    bacterial, and is a target for every other source. Omitting it made the ~40%
    identity boundary look source-specific when it is not.
  * It omitted epsilonproteobacteria (Campylobacterota) entirely, which made the
    8-against-6 retention separation read as False in all 12 arms when it is True
    in all 9 testable ones.

Ask for targets relative to a source with `targets_by_clade(source)`, never by
writing a literal set at the call site.
"""

BACTERIA = ("gammaproteobacteria", "firmicutes", "alphaproteobacteria",
            "betaproteobacteria", "actinobacteria", "cyanobacteria",
            "epsilonproteobacteria", "spirochaetes", "bacteroidetes")
ARCHAEA = ("crenarchaeota", "euryarchaeota")
EUKARYOTA = ("vertebrata", "streptophyta", "ascomycota", "insecta")

# 'other_bacteria' and 'other_eukaryota' are remainders, not clades, and are
# dropped everywhere in this project rather than being assigned here.
NOT_A_CLADE = ("other_bacteria", "other_eukaryota")

CLADE = {}
for _g in BACTERIA:
    CLADE[_g] = "bacteria"
for _g in ARCHAEA:
    CLADE[_g] = "archaea"
for _g in EUKARYOTA:
    CLADE[_g] = "eukaryota"

USABLE = tuple(sorted(CLADE))
assert len(USABLE) == 15, len(USABLE)


def clade_of(group):
    return CLADE.get(group)


def same_clade(a, b):
    return clade_of(a) is not None and clade_of(a) == clade_of(b)


def targets_by_clade(source, groups=None):
    """(within_clade, outside_clade) target lists for one source, source excluded.

    For a bacterial source this is (other bacteria, archaea + eukaryotes), which
    is the split the retention result is about. For a eukaryotic source it is
    (other eukaryotes, everything else) -- the same question asked from the other
    side, which is the only way the boundary claim can be tested off the
    gammaproteobacteria row.
    """
    pool = [g for g in (groups or USABLE) if g in CLADE and g != source]
    within = [g for g in pool if same_clade(g, source)]
    outside = [g for g in pool if not same_clade(g, source)]
    return within, outside
