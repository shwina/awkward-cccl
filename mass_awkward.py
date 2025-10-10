
import awkward as ak
import numpy as np
import cupy as cp
import uproot
import nvtx

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
mass = np.sqrt(
    2 * mu1.pt * mu2.pt * (np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi))
)
cp.cuda.Device().synchronize()

# benchmark run:
with nvtx.annotate("mass_calculations"):
    mass = np.sqrt(
        2 * mu1.pt * mu2.pt * (np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi))
    )
    cp.cuda.Device().synchronize()
