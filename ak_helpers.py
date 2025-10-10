import awkward as ak

def get_example1_buffers(array):
    form, length, buffers = ak.to_buffers(array)
    offsets_key = form.form_key
    data_key = form.content.form_key
    return buffers[f"{offsets_key}-offsets"], buffers[f"{data_key}-data"]
    
def get_example2_buffers(array):
    form, length, buffers = ak.to_buffers(array)
    offsets = buffers[f"{form.form_key}-offsets"]
    index   = buffers[f"{form.content.form_key}-index"]
    rec     = form.content.content
    data    = {fld: buffers[f"{c.form_key}-data"] for fld, c in zip(rec.fields, rec.contents)}
    return offsets, index, *(data.values())


def make_like_offsets(array, dtype="float32", kind="empty"):
    # Use buffers only; drop any Indexed layer; keep outer offsets as-is.
    form, length, bufs = ak.to_buffers(array)
    offsets = bufs[f"{form.form_key}-offsets"]        # top ListOffset offsets

    # choose array module by backend
    xp = __import__("cupy" if ak.backend(array) == "cuda" else "numpy")

    total = int(offsets[-1])                          # logical element count
    base  = xp.zeros(total, dtype=dtype) if kind == "zeros" else xp.empty(total, dtype=dtype)

    # Build a new ListOffset->Numpy form and container
    lo_key, leaf_key = "lo", "leaf"
    new_form = ak.forms.ListOffsetForm(
        form.offsets,                                 # preserve i32/i64 offsets type
        ak.forms.NumpyForm(dtype, form_key=leaf_key),
        form_key=lo_key,
    )
    new_bufs = {f"{lo_key}-offsets": offsets, f"{leaf_key}-data": base}
    return ak.from_buffers(new_form, length, new_bufs, backend="cuda")