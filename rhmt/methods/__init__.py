"""RHMT method registry."""
from rhmt.methods.baselines import ASR, MLSEM, PoCA
from rhmt.methods.rht_s import RHMTScatter
from rhmt.methods.rht_3p import RHMTScatter3
from rhmt.methods.rht_6p import RHMTSpectro
from rhmt.methods.asr_p import ASRMomentum

REGISTRY = {m.name: m for m in (PoCA, ASR, MLSEM, ASRMomentum, RHMTScatter3, RHMTScatter, RHMTSpectro)}


def get(name, dev="cuda:0"):
    return REGISTRY[name](dev=dev)
