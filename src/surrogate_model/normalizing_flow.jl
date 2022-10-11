module NormalizingFlow

using PyCall

function __init__()
    py"""

    import torch
    import torch.optim as optim
    import jammy_flows

    """
end
end