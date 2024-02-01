#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

# from mace.calculators import mace_mp
from ase.io import read, write
from ase import Atoms
from ase.neighborlist import neighbor_list as NL
from runschema.system import Descriptors, SOAP, MACE

from nomad.normalizing.normalizer import SystemBasedNormalizer


class AtomicDescriptorNormalizer(SystemBasedNormalizer):
    def normalize_system(self, system, is_representative):
        try:
            from quippy import descriptors
        except ImportError:
            descriptors = None

        # Only store SOAP for representative system to start with
        if not is_representative:
            return True

        # structures used to be stored in results, adding them back like this for now...
        if system.atoms is None:
            return False

        if not descriptors:
            self.logger.warning("SOAP normalizer runs, but quippy is not installed.")
            return False

        atoms = system.atoms.to_ase(raise_exp=True)

        # SOAP first
        soap = SOAP()
        # 1. scales the structure so that the average nearest neighbour distance matches diamond
        # 2. replaces all atoms with carbon so that descriptor is element agnostic
        # combination of 1. and 2. makes descriptors purely "structural"
        transformed_atoms = transform_struc(atoms)

        # setup params to be used by quippy
        # TODO decide whether to really store all of these OR to just include this in the descriptor of the SOAP descriptors itself...
        soap.n_max = 8
        soap.l_max = 4
        soap.r_cut = np.float64(5.0)
        soap.atom_sigma = np.float64(0.4)
        soap.R_mix = True
        soap.Z_mix = False
        soap.sym_mix = False
        soap.K = 8
        soap.coupling = False
        soap.radial_basis = "GTO"
        soap.average = True

        # compute the element agnostic, radially scaled SOAP descriptors
        quippy_str = f"soap cutoff={soap.r_cut} l_max={soap.l_max} n_max={soap.n_max} radial_basis={soap.radial_basis} atom_sigma={soap.atom_sigma} average={soap.average}"
        quippy_str += f" n_Z=1 Z={6} n_species=1 species_Z={6} R_mix={soap.R_mix} sym_mix={soap.sym_mix} Z_mix={soap.Z_mix} K={soap.K} coupling={soap.coupling}"

        desc = descriptors.Descriptor(quippy_str)
        # the final element is covariance_sigma0 which is always zero
        structural_soap_descriptor = desc.calc(transformed_atoms)["data"][0][:-1]
        soap.structural_soap = structural_soap_descriptor

        # add the soap descriptor to the system
        if not system.descriptors:
            system.descriptors = Descriptors()
        system.descriptors.soap = soap

        # now MACE

        try:
            from mace.calculators import mace_mp
        except ImportError:
            mace_mp = None

        if not mace_mp:
            self.logger.warning("Can't import MACE foundation model.")
            return False

        # load the calculator once only.
        if not hasattr(self, "mace_calculator"):
            print("jpd47 loading the mace calculator")
            self.mace_calculator = mace_mp(default_dtype="float64", dispersion=False)

        mace = MACE()
        mace_descriptors = self.mace_calculator.get_descriptors(atoms)
        mace.system_descriptor = np.mean(mace_descriptors, axis=0)
        if not system.descriptors:
            system.descriptors = Descriptors()
        system.descriptors.mace = mace

        return True


def transform_struc(struc, r_nn_targ=1.54):
    """1. finds the nearest neighbour distances for each atom in the structure
    2. scales the structure so that the average nearest neighbour distance is r_nn_targ, default is to match diamond
    3. converts all atoms to C
    """

    # find the nearest neighbour distances for each atom
    rcuts = [5, 10]
    for rcut in rcuts:
        i_list, ds = NL("id", struc, rcut, self_interaction=False)
        min_ds = [None for i in struc]
        for i, d in zip(i_list, ds):
            if min_ds[i] is None or d < min_ds[i]:
                min_ds[i] = d
        if not None in min_ds:
            break

    # compute the scale factor for the structure.
    f_nn = 1
    min_ds = [d for d in min_ds if d is not None]
    if len(min_ds) > 0:
        d_av = np.mean(min_ds)
        f_nn = (r_nn_targ) / d_av

    new_struc = Atoms(
        ["C" for Z in struc.numbers],
        scaled_positions=struc.get_scaled_positions(),
        cell=f_nn * np.array(struc.cell),
        pbc=True,
    )
    return new_struc
