import equinox as eqx


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
