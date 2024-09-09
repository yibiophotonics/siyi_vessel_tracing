
def CountParameters(model, only_trainable_parameters = False):
    """
    Count the number of parameters in a model.
    """
    if only_trainable_parameters:
        n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        n_parameters = sum([p.numel() for p in model.parameters()])
    return n_parameters