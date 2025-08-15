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
    
    # Test hopping operator
    hop = fermionic.FermionicHopping(0.5)
    assert hop.name == 'hop(0.5)'
    assert hop.dom == fermion @ fermion
    assert hop.cod == fermion @ fermion
    assert hop.t == 0.5


def test_fermionic_swap():
    """Test fermionic swap operation."""
    fermion = fermionic.fermion
    swap = fermionic.Swap(fermion, fermion)
    
    assert swap.dom == fermion @ fermion
    assert swap.cod == fermion @ fermion


def test_fermionic_gates_dict():
    """Test fermionic gates dictionary."""
    gates = fermionic.FERMIONIC_GATES
    
    assert 'c_dagger' in gates
    assert 'c' in gates
    assert 'hop' in gates
    
    # Test instantiation
    c_dagger = gates['c_dagger']
    assert isinstance(c_dagger, fermionic.FermionicCreation)
    
    c = gates['c']
    assert isinstance(c, fermionic.FermionicAnnihilation)
    
    hop_fn = gates['hop']
    assert callable(hop_fn)
    hop_instance = hop_fn(1.5)
    assert isinstance(hop_instance, fermionic.FermionicHopping)
    assert hop_instance.t == 1.5


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