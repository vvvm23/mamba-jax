import equinox as eqx


# Equinox layers don't have a dtype parameter, so we need to manually cast them
# TODO: any params we should filter from being cast along with other parameters in layer?
def cast_eqx_layer(layer, dtype):
    weight, bias = None, None
    if hasattr(layer, "weight") and layer.weight is not None:
        weight = layer.weight.astype(dtype)
    if hasattr(layer, "bias") and layer.bias is not None:
        bias = layer.bias.astype(dtype)

    if weight is not None:
        layer = eqx.tree_at(lambda l: l.weight, layer, weight)

    if bias is not None:
        layer = eqx.tree_at(lambda l: l.bias, layer, bias)

    return layer
