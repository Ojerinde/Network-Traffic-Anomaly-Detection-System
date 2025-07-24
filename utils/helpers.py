def flatten_nested_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_nested_dict(v))
        else:
            out[k] = v
    return out
