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
import json

from nomad.datamodel import EntryArchive
from atomicdescriptorsnormalizer import AtomicDescriptorNormalizer
import runschema  # pylint: disable=unused-import


def test_soap():
    archive = EntryArchive.m_from_dict(json.load(open("tests/data/vasp.archive.json")))

    AtomicDescriptorNormalizer(archive, only_representatives=True).normalize()

    #assert archive.run[-1].system[-1].descriptors.soap is not None
    #assert archive.run[-1].system[-1].descriptors.mace is not None
