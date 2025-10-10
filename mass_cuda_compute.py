
import awkward as ak
import cupy as cp
import numpy as np
import nvtx
import uproot

from cuda.compute import PermutationIterator, ZipIterator, gpu_struct, binary_transform

from ak_helpers import *

def compute_mass(mu1, mu2):
    offsets1, index1, pt1, eta1, phi1, charge1 = get_example2_buffers(mu1)
    offsets2, index2, pt2, eta2, phi2, charge2 = get_example2_buffers(mu2)

    d_in1 = PermutationIterator(ZipIterator(pt1, eta1, phi1, charge1), index1)
    d_in2 = PermutationIterator(ZipIterator(pt2, eta2, phi2, charge2), index2)

    mass = make_like_offsets(mu1.pt)
    d_out = mass.layout.content.data

    Muon = gpu_struct({
            "pt": np.float32,
            "eta": np.float32,
            "phi": np.float32,
            "charge": np.int32
        })

    def op(mu1: Muon, mu2: Muon) -> np.float32:
        return (
            2 * mu1.pt * mu2.pt * (np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi))
        )**0.5          

    # warm-up run:
    binary_transform(d_in1, d_in2, d_out, op, len(index1))
    cp.cuda.Device().synchronize()
    return mass

if __name__ == "__main__":     
    file = uproot.open(
        "https://github.com/jpivarski-talks/2023-12-18-hsf-india-tutorial-bhubaneswar/raw/main/data/SMHiggsToZZTo4L.root"
    )
    tree = file["Events"]
    arrays = tree.arrays(filter_name="/Muon_(pt|eta|phi|charge)/")
    muons = ak.zip(
        {
            "pt": arrays["Muon_pt"],
            "eta": arrays["Muon_eta"],
            "phi": arrays["Muon_phi"],
            "charge": arrays["Muon_charge"],
        }
    )
    muons = ak.concatenate([muons] * 100)
    
    pairs = ak.combinations(muons, 2)
    pairs = ak.to_backend(pairs, "cuda")
    
    mu1, mu2 = ak.unzip(pairs)
    
    # warmup run
    compute_mass(mu1, mu2)
    
    # benchmark run
    with nvtx.annotate("mass_calculation"):
        compute_mass(mu1, mu2)
