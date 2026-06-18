"""MDMT method registry."""
from mdmt.methods.baselines import ASR, MLSEM, PoCA
from mdmt.methods.rht_s import MDMTScatter
from mdmt.methods.rht_3p import MDMTScatter3
from mdmt.methods.rht_6p import MDMTSpectro
from mdmt.methods.asr_p import ASRMomentum
from mdmt.methods.poca_p import PoCAMomentum
from mdmt.methods.mlsem_p import MLSEMMomentum

REGISTRY = {m.name: m for m in (PoCA, ASR, MLSEM, ASRMomentum, PoCAMomentum,
                                MLSEMMomentum, MDMTScatter3, MDMTScatter, MDMTSpectro)}


def get(name, dev="cuda:0"):
    return REGISTRY[name](dev=dev)
