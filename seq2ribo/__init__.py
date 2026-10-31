"""seq2ribo: structure-aware simulation + Mamba polisher for ribosome A-site profiles."""

__version__ = "0.2.0"
__all__ = ["Seq2Ribo", "download_paper_weights", "RiboPolisherMamba", "MambaTEFull", "MambaTEFullUTR", "MambaExprFull"]


def __getattr__(name):
    # model classes need torch and mamba-ssm; import them only when requested so that the
    # simulation, geometry and metrics modules work in CPU-only environments
    if name in ("Seq2Ribo", "download_paper_weights"):
        from . import inference
        return getattr(inference, name)
    if name in ("RiboPolisherMamba", "MambaTEFull", "MambaTEFullUTR", "MambaExprFull"):
        from . import models
        return getattr(models, name)
    raise AttributeError(name)
