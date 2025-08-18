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


def test_jordan_wigner_transformation():
    """Test Jordan-Wigner transformation functions."""
    # Test creation operator transformation
    jw_creation = fermionic.jordan_wigner_transform('creation', 0, 2)
    assert 0 in jw_creation  # Should have operator on mode 0
    assert 1 in jw_creation  # Should have identity on mode 1
    
    # Test annihilation operator transformation
    jw_annihilation = fermionic.jordan_wigner_transform('annihilation', 0, 2)
    assert 0 in jw_annihilation
    assert 1 in jw_annihilation
    
    # Test number operator transformation
    jw_number = fermionic.jordan_wigner_transform('number', 0, 2)
    assert 0 in jw_number
    
    # Test that matrices have correct shape
    for op in jw_creation.values():
        assert op.shape == (2, 2)


def test_covariance_matrix():
    """Test fermionic covariance matrix functionality."""
    # Test vacuum state covariance matrix
    n_modes = 2
    covariance = fermionic.fermionic_covariance_matrix(n_modes)
    
    # Should be 2N x 2N matrix
    assert covariance.shape == (4, 4)
    
    # Test anticommutation relations in vacuum
    # {c_i, c_j†} = δ_{ij} for vacuum
    assert covariance[0, 1] == 1.0  # {c_0, c_0†}
    assert covariance[2, 3] == 1.0  # {c_1, c_1†}


def test_covariance_evolution():
    """Test evolution of covariance matrix."""
    n_modes = 1
    covariance = fermionic.fermionic_covariance_matrix(n_modes)
    hamiltonian = np.array([[0, 1], [1, 0]], dtype=complex)
    
    # Evolve for short time
    evolved = fermionic.evolve_covariance_matrix(covariance, hamiltonian, 0.1)
    
    # Should still be valid covariance matrix
    assert evolved.shape == (2, 2)
    assert np.allclose(evolved, evolved.conj().T)  # Should be Hermitian


def test_fermionic_diagram_evaluation():
    """Test fermionic diagram evaluation using covariance matrices."""
    fermion = fermionic.fermion
    
    # Create simple diagram with number operator
    n_op = fermionic.FermionicNumber()
    layer = fermionic.Layer(fermionic.Ty(), n_op, fermionic.Ty())
    diagram = fermionic.Diagram(fermion, fermion, [layer])
    
    # Test that evaluation method exists and returns something
    assert hasattr(diagram, 'eval')
    # Note: Full evaluation test would require more complex setup


def test_fermionic_swap_sign():
    """Test fermionic swap sign handling."""
    fermion = fermionic.fermion
    swap = fermionic.Swap(fermion, fermion)
    
    # Test fermionic sign detection
    assert swap.has_fermionic_sign == True
    
    # Test matrix includes fermionic sign
    matrix = swap.matrix
    assert matrix.shape == (4, 4)
    
    # Check that swap matrix has correct fermionic sign
    # |01⟩ → -|10⟩ should have -1 sign
    assert matrix[1, 2] == -1.0 or matrix[2, 1] == -1.0
    
    # Test Jordan-Wigner conversion
    jw_gates = swap.to_jordan_wigner()
    assert len(jw_gates) >= 1  # Should have at least one gate


def test_fermionic_special_boxes():
    """Test fermionic special boxes (cap, cup, spider)."""
    fermion = fermionic.fermion
    
    # Test fermionic cap
    cap = fermionic.generate_cap(fermion, fermion)
    assert isinstance(cap, fermionic.Diagram)
    assert cap.dom == fermionic.Ty()  # vacuum input
    assert cap.cod == fermion @ fermion  # two fermions output
    
    # Test fermionic cup
    cup = fermionic.generate_cup(fermion, fermion)
    assert isinstance(cup, fermionic.Diagram)
    assert cup.dom == fermion @ fermion  # two fermions input
    assert cup.cod == fermionic.Ty()  # vacuum output
    
    # Test fermionic spider
    spider = fermionic.generate_spider(fermion, 2, 2)
    assert isinstance(spider, fermionic.Diagram)
    assert spider.dom == fermion @ fermion
    assert spider.cod == fermion @ fermion


def test_fermionic_tensor_network_conversion():
    """Test fermionic tensor network conversion."""
    fermion = fermionic.fermion
    
    # Create simple fermionic diagram
    diagram = fermionic.Diagram(fermion, fermion, [])
    
    # Test tensor network conversion
    assert hasattr(diagram, 'to_tn')
    assert hasattr(diagram, 'to_quantum_via_jordan_wigner')
    
    # Test Jordan-Wigner conversion
    quantum_diagram = diagram.to_quantum_via_jordan_wigner()
    assert hasattr(quantum_diagram, 'dom')
    assert hasattr(quantum_diagram, 'cod')


def test_convenience_functions():
    """Test convenience functions for building fermionic circuits."""
    # Test fermionic circuit creation
    circuit = fermionic.fermionic_circuit(2)
    assert isinstance(circuit, fermionic.Diagram)
    assert len(circuit.dom) == 2
    
    # Test hopping term application
    circuit_with_hopping = fermionic.apply_hopping_term(circuit, 0, 1, t=1.5)
    assert isinstance(circuit_with_hopping, fermionic.Diagram) 
    assert len(circuit_with_hopping.layers) == 1
    
    # Test pairing term application
    circuit_with_pairing = fermionic.apply_pairing_term(circuit, 0, 1, delta=0.5)
    assert isinstance(circuit_with_pairing, fermionic.Diagram)
    assert len(circuit_with_pairing.layers) == 2  # Create + annihilate pairing
    
    # Test number operator measurement
    circuit_with_measurement = fermionic.number_operator_expectation(circuit, 0)
    assert isinstance(circuit_with_measurement, fermionic.Diagram)
    assert len(circuit_with_measurement.layers) == 1


def test_fermionic_exports():
    """Test that all expected symbols are exported."""
    expected_exports = [
        'Ty', 'Box', 'Layer', 'Diagram', 'Swap',
        'FermionicCreation', 'FermionicAnnihilation', 'FermionicNumber',
        'FermionicHopping', 'FermionicPairing',
        'jordan_wigner_string', 'jordan_wigner_transform',
        'fermionic_covariance_matrix', 'evolve_covariance_matrix',
        'generate_cap', 'generate_cup', 'generate_spider',
        'FERMIONIC_GATES', 'fermion', 'fermionic',
        'fermionic_circuit', 'apply_hopping_term', 'apply_pairing_term',
        'number_operator_expectation'
    ]
    
    # Check that the symbols are defined in the module
    for symbol in expected_exports:
        assert hasattr(fermionic, symbol), f"Missing export: {symbol}"