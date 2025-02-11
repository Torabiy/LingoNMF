import mir_eval

def evaluate_separation(X, XH):
    """
    Evaluates source separation using SDR, SIR, and SAR.
    """
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(X, XH)
    return sdr, sir, sar
