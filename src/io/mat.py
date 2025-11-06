from scipy.io import loadmat, savemat

def load_mat(path):
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k:v for k,v in raw.items() if not k.startswith("__")}

def save_mat(path, **kwargs):
    savemat(path, kwargs)
