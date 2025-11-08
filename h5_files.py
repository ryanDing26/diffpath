import h5py

h5_path = "/shares/sinha/sinha_common/GTEx/h5-tiles-256/GTEX-15UF7-1626.h5"

with h5py.File(h5_path, 'r') as f:
    for key, val in f.items():
        print(f"{key}: {type(val)}")
        if isinstance(val, h5py.Dataset):
            print("  shape:", val.shape)
            print("  dtype:", val.dtype)
