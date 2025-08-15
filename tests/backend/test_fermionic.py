# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Test module for fermionic backend.
"""

import pytest
import numpy as np

from lambeq.backend import fermionic


def test_fermionic_type():
    """Test basic fermionic type creation."""
    fermion = fermionic.fermion
    assert str(fermion) == 'fermion'
    assert repr(fermion) == 'Ty(fermion)'
    
    # Test composition
    two_fermions = fermion @ fermion
    assert len(two_fermions) == 2


def test_fermionic_diagram():
    """Test basic fermionic diagram creation."""
    fermion = fermionic.fermion
    diagram = fermionic.Diagram(fermion, fermion, [])
    
    assert diagram.dom == fermion
    assert diagram.cod == fermion
    assert len(diagram.layers) == 0


def test_fermionic_boxes():
    """Test fermionic box creation."""
    fermion = fermionic.fermion
    
    # Test creation operator
    c_dagger = fermionic.FermionicCreation()
    assert c_dagger.name == 'c†'
    assert c_dagger.dom == fermionic.Ty()  # vacuum
    assert c_dagger.cod == fermion
    
    # Test annihilation operator  
    c = fermionic.FermionicAnnihilation()
    assert c.name == 'c'
    assert c.dom == fermion
    assert c.cod == fermionic.Ty()  # vacuum
    
    # Test number operator
    n = fermionic.FermionicNumber()
    assert n.name == 'n'
    assert n.dom == fermion
    assert n.cod == fermion
    
    # Test hopping operator
    hop = fermionic.FermionicHopping(0.5)
    assert hop.name == 'hop(0.5)'
    assert hop.dom == fermion @ fermion
    assert hop.cod == fermion @ fermion
    assert hop.t == 0.5
    
    # Test pairing operators
    delta_dagger = fermionic.FermionicPairing(creation=True)
    assert delta_dagger.name == 'Δ†'
    assert delta_dagger.dom == fermionic.Ty()  # vacuum
    assert delta_dagger.cod == fermion @ fermion
    assert delta_dagger.creation == True
    
    delta = fermionic.FermionicPairing(creation=False)
    assert delta.name == 'Δ'
    assert delta.dom == fermion @ fermion
    assert delta.cod == fermionic.Ty()  # vacuum
    assert delta.creation == False


def test_fermionic_swap():
    """Test fermionic swap operation."""
    fermion = fermionic.fermion
    swap = fermionic.Swap(fermion, fermion)
    
    assert swap.dom == fermion @ fermion
    assert swap.cod == fermion @ fermion
    
    # Test fermionic sign property
    assert swap.has_fermionic_sign == True
    
    # Test non-fermionic swap
    qubit = fermionic.quantum.qubit
    qubit_swap = fermionic.Swap(qubit, qubit)
    assert qubit_swap.has_fermionic_sign == False


def test_fermionic_gates_dict():
    """Test fermionic gates dictionary."""
    gates = fermionic.FERMIONIC_GATES
    
    assert 'c_dagger' in gates
    assert 'c' in gates
    assert 'n' in gates
    assert 'hop' in gates
    assert 'delta_dagger' in gates
    assert 'delta' in gates
    
    # Test instantiation
    c_dagger = gates['c_dagger']
    assert isinstance(c_dagger, fermionic.FermionicCreation)
    
    c = gates['c']
    assert isinstance(c, fermionic.FermionicAnnihilation)
    
    n = gates['n']
    assert isinstance(n, fermionic.FermionicNumber)
    
    hop_fn = gates['hop']
    assert callable(hop_fn)
    hop_instance = hop_fn(1.5)
    assert isinstance(hop_instance, fermionic.FermionicHopping)
    assert hop_instance.t == 1.5
    
    # Test pairing operators
    delta_dagger_fn = gates['delta_dagger']
    delta_dagger = delta_dagger_fn()
    assert isinstance(delta_dagger, fermionic.FermionicPairing)
    assert delta_dagger.creation == True
    
    delta_fn = gates['delta']
    delta = delta_fn()
    assert isinstance(delta, fermionic.FermionicPairing)
    assert delta.creation == False


def test_fermionic_inheritance():
    """Test that fermionic classes properly inherit from quantum classes."""
    fermion = fermionic.fermion
    
    # Test Ty inheritance
    assert issubclass(fermionic.Ty, fermionic.quantum.Ty)
    
    # Test Box inheritance
    assert issubclass(fermionic.Box, fermionic.quantum.Box)
    
    # Test Diagram inheritance
    assert issubclass(fermionic.Diagram, fermionic.quantum.Diagram)
    
    # Test Layer inheritance
    assert issubclass(fermionic.Layer, fermionic.quantum.Layer)


def test_fermionic_conversion_methods():
    """Test that fermionic diagrams have conversion methods."""
    fermion = fermionic.fermion
    diagram = fermionic.Diagram(fermion, fermion, [])
    
    # Check methods exist
    assert hasattr(diagram, 'eval')
    assert hasattr(diagram, 'to_tn') 
    assert hasattr(diagram, 'to_tk')
    assert hasattr(diagram, 'to_pennylane')
    
    # Test they are callable
    assert callable(diagram.eval)
    assert callable(diagram.to_tn)
    assert callable(diagram.to_tk)  
    assert callable(diagram.to_pennylane)


def test_fermionic_special_boxes_registration():
    """Test that fermionic special boxes are properly registered."""
    fermion = fermionic.fermion
    
    # Test that special box generators exist
    assert hasattr(fermionic, 'generate_cap')
    assert hasattr(fermionic, 'generate_cup')
    assert hasattr(fermionic, 'generate_spider')
    
    # Test they are callable
    assert callable(fermionic.generate_cap)
    assert callable(fermionic.generate_cup)
    assert callable(fermionic.generate_spider)


def test_jordan_wigner_string():
    """Test Jordan-Wigner string helper function."""
    # Test adjacent modes (no string needed)
    assert fermionic.jordan_wigner_string(0, 0) == "I"
    assert fermionic.jordan_wigner_string(0, 1) == "I" 
    assert fermionic.jordan_wigner_string(1, 0) == "I"
    
    # Test distant modes (string needed)
    assert fermionic.jordan_wigner_string(0, 2) == "Z^⊗1"
    assert fermionic.jordan_wigner_string(0, 3) == "Z^⊗2"
    assert fermionic.jordan_wigner_string(1, 4) == "Z^⊗2"
    assert fermionic.jordan_wigner_string(4, 1) == "Z^⊗2"


def test_fermionic_example():
    """Test a simple fermionic circuit example."""
    fermion = fermionic.fermion
    
    # Create fermionic operators
    c_dagger = fermionic.FermionicCreation()
    c = fermionic.FermionicAnnihilation()
    n = fermionic.FermionicNumber()
    
    # Test that we can create basic fermionic circuits
    # This is a conceptual test - in a full implementation these would
    # be composed into actual fermionic diagrams
    assert c_dagger.cod == fermion  # creation gives fermion
    assert c.dom == fermion         # annihilation takes fermion  
    assert n.dom == n.cod == fermion  # number preserves fermion