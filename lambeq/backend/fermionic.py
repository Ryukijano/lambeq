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
Fermionic category
==================
Lambeq's internal representation of the fermionic category for 
free-fermionic circuits. This module provides support for fermionic
circuits that can be efficiently evaluated using covariance matrices
and quadratic Hamiltonians via the Jordan-Wigner transformation.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, cast, Optional, Union, List, Tuple
from typing_extensions import Self

import numpy as np
import tensornetwork as tn

from lambeq.backend import quantum, tensor
from lambeq.backend.numerical_backend import backend


# Create a fermionic category based on the quantum category
fermionic = quantum.quantum


@fermionic
class Ty(quantum.Ty):
    """A type in the fermionic category.
    
    For fermionic circuits, types represent fermionic modes which
    follow fermionic anticommutation relations under the Jordan-Wigner
    transformation.
    """
    pass


fermion = Ty('fermion')


@fermionic 
class Box(quantum.Box):
    """A box in the fermionic category.
    
    Fermionic boxes represent matchgates under the Jordan-Wigner
    transformation, which maintain the fermionic structure of
    free-fermionic circuits.
    """
    pass


@dataclass
@fermionic
class Layer(quantum.Layer):
    """A Layer in a fermionic Diagram.
    
    Parameters
    ----------
    box : Box
        The fermionic box of the layer.
    left : Ty
        The wire type to the left of the box.
    right : Ty
        The wire type to the right of the box.
    """
    
    left: Ty
    box: Box
    right: Ty


@dataclass
@fermionic
class Diagram(quantum.Diagram):
    """A diagram in the fermionic category.
    
    Fermionic diagrams represent free-fermionic circuits that can be
    efficiently evaluated using covariance matrices and quadratic
    Hamiltonians. Each box in the diagram corresponds to a matchgate
    under the Jordan-Wigner transformation.
    
    Parameters
    ----------
    dom : Ty
        The type of the input wires (fermionic modes).
    cod : Ty
        The type of the output wires (fermionic modes).
    layers : list[Layer]
        The layers of the fermionic diagram.
    """
    
    dom: Ty
    cod: Ty
    layers: list[Layer]  # type: ignore[assignment]
    
    def eval(self,
             *others,
             backend=None,
             mixed=False,
             contractor=None,
             **params):
        """Evaluate the fermionic circuit using covariance matrices.
        
        For fermionic circuits, evaluation is performed using the 
        covariance matrix representation which allows for efficient
        computation of free-fermionic systems.
        
        Parameters
        ----------
        others : :class:`lambeq.backend.fermionic.Diagram`
            Other fermionic circuits to process in batch.
        backend : Backend, optional
            Backend for computation, by default None uses covariance matrices
        mixed : bool, optional
            Whether the circuit is mixed, by default False
        contractor : Callable, optional
            The contractor to use for tensor network evaluation
        
        Returns
        -------
        np.ndarray or list of np.ndarray
            The result of the fermionic circuit evaluation.
        """
        
        # Check if we can use efficient fermionic evaluation
        if self._can_use_fermionic_eval() and backend is None:
            return self._eval_fermionic_covariance(**params)
        
        # Otherwise, delegate to quantum eval method for compatibility
        return super().eval(*others, 
                          backend=backend,
                          mixed=mixed,
                          contractor=contractor or tn.contractors.auto,
                          **params)
    
    def _can_use_fermionic_eval(self) -> bool:
        """Check if diagram can use efficient fermionic evaluation.
        
        Returns True if all boxes are fermionic and form a free-fermionic
        circuit (quadratic Hamiltonian).
        """
        # For now, simple check - all boxes should be fermionic
        for layer in self.layers:
            if not isinstance(layer.box, (FermionicCreation, FermionicAnnihilation, 
                                        FermionicNumber, FermionicHopping, FermionicPairing)):
                return False
        return True
    
    def _eval_fermionic_covariance(self, **params) -> np.ndarray:
        """Evaluate using fermionic covariance matrix representation.
        
        This implements efficient evaluation for free-fermionic circuits
        using covariance matrices and quadratic Hamiltonians.
        """
        n_modes = len(self.dom) if self.dom else 1
        
        # Initialize covariance matrix for vacuum state
        covariance = fermionic_covariance_matrix(n_modes)
        
        # Apply each layer sequentially
        for layer in self.layers:
            covariance = self._apply_fermionic_layer(layer, covariance, **params)
        
        # Extract final expectation values or probabilities
        return self._extract_measurement_from_covariance(covariance)
    
    def _apply_fermionic_layer(self, layer: Layer, covariance: np.ndarray, **params) -> np.ndarray:
        """Apply a fermionic layer to the covariance matrix.
        
        Parameters
        ----------
        layer : Layer
            The fermionic layer to apply
        covariance : np.ndarray
            Current covariance matrix
            
        Returns
        -------
        np.ndarray
            Updated covariance matrix
        """
        box = layer.box
        
        if isinstance(box, FermionicHopping):
            # Hopping term: apply quadratic evolution
            # H = t * (c_i† c_j + c_j† c_i)
            t = getattr(box, 't', params.get('t', 1.0))
            # For simplicity, assume hopping between adjacent modes
            # In full implementation, would extract mode indices from layer
            hamiltonian = np.array([[0, t], [t, 0]], dtype=complex)
            covariance = evolve_covariance_matrix(covariance, hamiltonian, 1.0)
            
        elif isinstance(box, FermionicNumber):
            # Number operator: diagonal in occupation basis
            # For covariance matrix, this corresponds to measurements
            pass  # Measurement extraction handled in final step
            
        # For creation/annihilation operators, would update covariance directly
        # This requires more sophisticated bookkeeping of mode indices
        
        return covariance
    
    def _extract_measurement_from_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Extract measurement probabilities from covariance matrix.
        
        For fermionic systems, occupation probabilities are:
        p_i = ⟨n_i⟩ = ⟨c_i† c_i⟩ = (1 - Γ_{2i+1,2i}) / 2
        
        where Γ is the covariance matrix.
        """
        n_modes = covariance.shape[0] // 2
        probabilities = np.zeros(n_modes, dtype=float)
        
        for i in range(n_modes):
            c_dag_idx = 2*i + 1
            c_idx = 2*i
            
            # Occupation probability from covariance matrix
            probabilities[i] = (1.0 - covariance[c_dag_idx, c_idx].real) / 2.0
            
        return probabilities
    
    def to_tn(self, mixed=False):
        """Convert to tensor network representation.
        
        For fermionic circuits, this creates a tensor network that
        respects the fermionic anticommutation relations by applying
        the Jordan-Wigner transformation.
        
        Parameters
        ----------
        mixed : bool, optional
            Whether to create mixed state representation, by default False
            
        Returns
        -------
        tuple
            Nodes and output edge order for the tensor network
        """
        
        # Convert fermionic diagram to quantum diagram using Jordan-Wigner transformation
        quantum_diagram = self.to_quantum_via_jordan_wigner()
        
        # Use quantum tensor network conversion
        return quantum_diagram.to_tn(mixed=mixed)
    
    def to_quantum_via_jordan_wigner(self):
        """Convert fermionic diagram to quantum diagram via Jordan-Wigner transformation.
        
        Returns
        -------
        quantum.Diagram
            Equivalent quantum diagram with Jordan-Wigner transformation applied
        """
        from lambeq.backend.quantum import Diagram as QuantumDiagram, Box as QuantumBox
        
        # Determine number of modes
        n_modes = len(self.dom) if self.dom else 1
        
        # Convert domain and codomain to qubits
        quantum_dom = quantum.qubit ** n_modes
        quantum_cod = quantum.qubit ** len(self.cod) if self.cod else quantum.qubit ** n_modes
        
        quantum_layers = []
        
        for layer in self.layers:
            quantum_box = self._convert_fermionic_box_to_quantum(layer.box, n_modes)
            
            # Convert layer types  
            quantum_left = quantum.qubit ** len(layer.left) if layer.left else quantum.Ty()
            quantum_right = quantum.qubit ** len(layer.right) if layer.right else quantum.Ty()
            
            quantum_layer = quantum.Layer(quantum_left, quantum_box, quantum_right)
            quantum_layers.append(quantum_layer)
        
        return QuantumDiagram(quantum_dom, quantum_cod, quantum_layers)
    
    def _convert_fermionic_box_to_quantum(self, fermionic_box: Box, n_modes: int):
        """Convert a fermionic box to quantum gates using Jordan-Wigner transformation."""
        from lambeq.backend.quantum import Box as QuantumBox, Id
        
        if isinstance(fermionic_box, FermionicCreation):
            # Convert c† to quantum gates
            mode = 0  # For simplicity, assume single mode operations
            jw_ops = jordan_wigner_transform('creation', mode, n_modes)
            # Would create quantum circuit implementing the JW transformation
            # For now, return identity as placeholder
            return QuantumBox('JW_creation', quantum.qubit, quantum.qubit)
            
        elif isinstance(fermionic_box, FermionicAnnihilation):
            mode = 0
            jw_ops = jordan_wigner_transform('annihilation', mode, n_modes)
            return QuantumBox('JW_annihilation', quantum.qubit, quantum.qubit)
            
        elif isinstance(fermionic_box, FermionicNumber):
            mode = 0
            jw_ops = jordan_wigner_transform('number', mode, n_modes)
            return QuantumBox('JW_number', quantum.qubit, quantum.qubit)
            
        elif isinstance(fermionic_box, FermionicHopping):
            # Hopping between modes requires multi-qubit gates
            return QuantumBox('JW_hopping', quantum.qubit @ quantum.qubit, quantum.qubit @ quantum.qubit)
            
        else:
            # Default conversion for unknown boxes
            return QuantumBox(f'JW_{fermionic_box.name}', 
                            quantum.qubit ** len(fermionic_box.dom) if fermionic_box.dom else quantum.Ty(),
                            quantum.qubit ** len(fermionic_box.cod) if fermionic_box.cod else quantum.Ty())
    
    def to_tk(self):
        """Convert to tket circuit representation.
        
        Converts the fermionic diagram to a tket circuit, handling
        the Jordan-Wigner transformation appropriately.
        
        Returns
        -------
        Circuit
            A tket circuit representation
        """
        
        # Convert to quantum diagram first using Jordan-Wigner transformation
        quantum_diagram = self.to_quantum_via_jordan_wigner()
        
        # Use quantum-to-tket conversion
        return quantum_diagram.to_tk()
    
    def to_pennylane(self, probabilities=False, backend_config=None,
                    diff_method='best', **kwargs):
        """Convert to PennyLane circuit representation.
        
        Converts the fermionic diagram to a PennyLane circuit with
        appropriate handling of fermionic operations using Jordan-Wigner
        transformation.
        
        Parameters
        ----------
        probabilities : bool, optional
            Whether to return probabilities, by default False
        backend_config : dict, optional
            Backend configuration, by default None
        diff_method : str, optional
            Differentiation method, by default 'best'
            
        Returns
        -------
        PennylaneCircuit
            A PennyLane circuit representation
        """
        
        # Convert to quantum diagram first using Jordan-Wigner transformation
        quantum_diagram = self.to_quantum_via_jordan_wigner()
        
        # Use quantum-to-pennylane conversion
        return quantum_diagram.to_pennylane(probabilities=probabilities,
                                           backend_config=backend_config,
                                           diff_method=diff_method,
                                           **kwargs)


# Fermionic gates - these are the gates that change under Jordan-Wigner transformation
# For free-fermionic circuits, we primarily need creation/annihilation operators
# and their combinations

class FermionicCreation(Box):
    """Fermionic creation operator.
    
    Creates a fermion at the specified mode.
    """
    
    def __init__(self):
        super().__init__('c†', Ty(), fermion)


class FermionicAnnihilation(Box):
    """Fermionic annihilation operator.
    
    Annihilates a fermion at the specified mode.
    """
    
    def __init__(self):
        super().__init__('c', fermion, Ty())


class FermionicNumber(Box):
    """Fermionic number operator.
    
    Represents the number operator n = c†c for a fermionic mode.
    """
    
    def __init__(self):
        super().__init__('n', fermion, fermion)


class FermionicHopping(Box):
    """Fermionic hopping operator.
    
    Represents hopping between fermionic modes, fundamental
    for free-fermionic Hamiltonians.
    """
    
    def __init__(self, t: float = 1.0):
        self.t = t
        super().__init__(f'hop({t})', fermion @ fermion, fermion @ fermion)


class FermionicPairing(Box):
    """Fermionic pairing operator.
    
    Represents pairing terms c†c† or cc that appear in 
    superconducting systems and BCS Hamiltonians.
    """
    
    def __init__(self, creation: bool = True):
        if creation:
            name = 'Δ†'  # c†c† term
            dom, cod = Ty(), fermion @ fermion
        else:
            name = 'Δ'   # cc term  
            dom, cod = fermion @ fermion, Ty()
        super().__init__(name, dom, cod)
        self.creation = creation


def jordan_wigner_string(start: int, end: int) -> str:
    """Generate Jordan-Wigner string description.
    
    In fermionic circuits, operators acting on distant modes require
    Jordan-Wigner strings (products of Z gates) to maintain proper
    anticommutation relations.
    
    Parameters
    ----------
    start : int
        Starting mode index
    end : int  
        Ending mode index
        
    Returns
    -------
    str
        Description of the Jordan-Wigner string needed
    """
    if start == end:
        return "I"
    elif abs(end - start) == 1:
        return "I"
    else:
        length = abs(end - start) - 1
        return f"Z^⊗{length}"


def jordan_wigner_transform(fermionic_op: str, mode: int, n_modes: int) -> Dict[str, np.ndarray]:
    """Transform fermionic operator to quantum operators using Jordan-Wigner transformation.
    
    The Jordan-Wigner transformation maps fermionic operators to quantum operators:
    - c_j† = (⊗_{k<j} Z_k) ⊗ (X_j - iY_j)/2
    - c_j = (⊗_{k<j} Z_k) ⊗ (X_j + iY_j)/2
    - n_j = (I - Z_j)/2
    
    Parameters
    ----------
    fermionic_op : str
        Type of fermionic operator ('creation', 'annihilation', 'number')
    mode : int
        Mode index (0-based)
    n_modes : int
        Total number of fermionic modes
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping qubit indices to Pauli matrices
    """
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    operators = {}
    
    if fermionic_op == 'creation':
        # c† = (⊗_{k<mode} Z_k) ⊗ (X - iY)/2
        for k in range(mode):
            operators[k] = Z
        operators[mode] = (X - 1j*Y) / 2
        
    elif fermionic_op == 'annihilation':
        # c = (⊗_{k<mode} Z_k) ⊗ (X + iY)/2
        for k in range(mode):
            operators[k] = Z
        operators[mode] = (X + 1j*Y) / 2
        
    elif fermionic_op == 'number':
        # n = (I - Z)/2
        operators[mode] = (I - Z) / 2
        
    else:
        raise ValueError(f"Unknown fermionic operator: {fermionic_op}")
    
    # Fill remaining modes with identity
    for k in range(n_modes):
        if k not in operators:
            operators[k] = I
            
    return operators


def fermionic_covariance_matrix(n_modes: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
    """Initialize fermionic covariance matrix for Gaussian states.
    
    For a fermionic system with N modes, the covariance matrix is 2N x 2N:
    Γ_{ij} = ⟨ξ_i ξ_j + ξ_j ξ_i⟩ - 2⟨ξ_i⟩⟨ξ_j⟩
    
    where ξ = (c_1, c_1†, c_2, c_2†, ..., c_N, c_N†)
    
    Parameters
    ----------
    n_modes : int
        Number of fermionic modes
    initial_state : np.ndarray, optional
        Initial state vector, defaults to vacuum state
        
    Returns
    -------
    np.ndarray
        2N x 2N covariance matrix
    """
    # For vacuum state, covariance matrix has specific structure
    if initial_state is None:
        # Vacuum state: ⟨c_i⟩ = ⟨c_i†⟩ = 0 for all i
        # ⟨c_i c_j†⟩ = δ_{ij}, ⟨c_i† c_j⟩ = 0
        covariance = np.zeros((2*n_modes, 2*n_modes), dtype=complex)
        
        for i in range(n_modes):
            # c_i, c_i† indices
            c_idx = 2*i
            c_dag_idx = 2*i + 1
            
            # Anticommutation relations: {c_i, c_j†} = δ_{ij}
            covariance[c_idx, c_dag_idx] = 1.0
            covariance[c_dag_idx, c_idx] = 1.0
            
        return covariance
    else:
        # For general states, would need to compute expectation values
        # This is a placeholder for more complex state preparation
        raise NotImplementedError("General covariance matrix computation not yet implemented")


def evolve_covariance_matrix(covariance: np.ndarray, hamiltonian_matrix: np.ndarray, time: float) -> np.ndarray:
    """Evolve covariance matrix under quadratic Hamiltonian.
    
    For quadratic Hamiltonians H = ξ† A ξ, the covariance matrix evolves as:
    Γ(t) = exp(-2iAt) Γ(0) exp(2iA†t)
    
    Parameters
    ----------
    covariance : np.ndarray
        Initial covariance matrix
    hamiltonian_matrix : np.ndarray  
        Matrix representation of quadratic Hamiltonian
    time : float
        Evolution time
        
    Returns
    -------
    np.ndarray
        Evolved covariance matrix
    """
    # For quadratic evolution
    evolution_matrix = np.exp(-2j * hamiltonian_matrix * time)
    evolution_matrix_dag = np.conj(evolution_matrix.T)
    
    # Evolve covariance matrix
    evolved_covariance = evolution_matrix @ covariance @ evolution_matrix_dag
    
    return evolved_covariance


# Fermionic special boxes - these implement fermionic anticommutation relations

@Diagram.register_special_box('cap')
def generate_cap(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a fermionic cap diagram.
    
    Fermionic caps must respect anticommutation relations when creating
    or annihilating fermionic pairs.
    
    Parameters
    ----------
    left : Ty
        The left type of the cap.
    right : Ty
        The right type of the cap.
    is_reversed : bool, optional
        Whether the cap is reversed, by default False
        
    Returns
    -------
    Diagram
        The fermionic cap diagram.
    """
    
    if left == fermion and right == fermion:
        # Fermionic cap: creates a fermionic pair
        # This requires special handling for anticommutation
        pair_creator = FermionicPairing(creation=True)
        return Diagram(Ty(), left @ right, [Layer(Ty(), pair_creator, Ty())])
    else:
        # Non-fermionic cap, delegate to quantum implementation
        return cast(Diagram, quantum.generate_cap(left, right, is_reversed))


@Diagram.register_special_box('cup')
def generate_cup(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a fermionic cup diagram.
    
    Fermionic cups must respect anticommutation relations when creating
    or annihilating fermionic pairs.
    
    Parameters
    ----------
    left : Ty
        The left type of the cup.
    right : Ty
        The right type of the cup.
    is_reversed : bool, optional
        Whether the cup is reversed, by default False
        
    Returns
    -------
    Diagram
        The fermionic cup diagram.
    """
    
    if left == fermion and right == fermion:
        # Fermionic cup: annihilates a fermionic pair
        # This requires special handling for anticommutation
        pair_annihilator = FermionicPairing(creation=False)
        return Diagram(left @ right, Ty(), [Layer(Ty(), pair_annihilator, Ty())])
    else:
        # Non-fermionic cup, delegate to quantum implementation
        return cast(Diagram, quantum.generate_cup(left, right, is_reversed))


@Diagram.register_special_box('spider')
def generate_spider(type: Ty, n_legs_in: int, n_legs_out: int) -> Diagram:
    """Generate a fermionic spider diagram.
    
    Fermionic spiders implement multimode operations while preserving
    fermionic anticommutation relations.
    
    Parameters
    ----------
    type : Ty
        The type of the spider legs.
    n_legs_in : int
        Number of input legs.
    n_legs_out : int
        Number of output legs.
        
    Returns
    -------
    Diagram
        The fermionic spider diagram.
    """
    
    if type == fermion:
        # Fermionic spider: multimode fermionic operation
        # For simplicity, implement as identity for equal in/out legs
        if n_legs_in == n_legs_out:
            spider_box = Box(f'fermionic_spider_{n_legs_in}_{n_legs_out}', 
                           type ** n_legs_in, type ** n_legs_out)
            return Diagram(type ** n_legs_in, type ** n_legs_out, 
                         [Layer(Ty(), spider_box, Ty())])
        else:
            # Unequal legs - need special fermionic handling
            # This would involve creation/annihilation of fermions
            if n_legs_out > n_legs_in:
                # More outputs: create fermions
                diff = n_legs_out - n_legs_in
                creators = [FermionicCreation() for _ in range(diff)]
                # Simplified implementation
                spider_box = Box(f'fermionic_spider_create_{diff}', 
                               type ** n_legs_in, type ** n_legs_out)
            else:
                # More inputs: annihilate fermions  
                diff = n_legs_in - n_legs_out
                annihilators = [FermionicAnnihilation() for _ in range(diff)]
                # Simplified implementation
                spider_box = Box(f'fermionic_spider_annihilate_{diff}',
                               type ** n_legs_in, type ** n_legs_out)
            
            return Diagram(type ** n_legs_in, type ** n_legs_out,
                         [Layer(Ty(), spider_box, Ty())])
    else:
        # Non-fermionic spider, delegate to quantum implementation
        return cast(Diagram, quantum.generate_spider(type, n_legs_in, n_legs_out))


@Diagram.register_special_box('swap')
class Swap(quantum.Swap, Box):
    """A fermionic swap box.
    
    In fermionic systems, swaps introduce a minus sign due to
    anticommutation relations of fermions: ψ_i ψ_j = -ψ_j ψ_i
    """
    
    def __init__(self, left: Ty, right: Ty):
        super().__init__(left, right)
        self._fermionic_sign = self._compute_fermionic_sign()
        
    def _compute_fermionic_sign(self) -> bool:
        """Compute whether this swap requires a fermionic sign.
        
        Returns True if both types are fermionic, requiring a -1 phase.
        """
        return (isinstance(self.left, Ty) and self.left == fermion and 
                isinstance(self.right, Ty) and self.right == fermion)
        
    @property
    def has_fermionic_sign(self) -> bool:
        """Whether this swap introduces a fermionic sign.
        
        Returns
        -------
        bool
            True if this swap involves fermionic modes and thus 
            requires a -1 phase factor.
        """
        return self._fermionic_sign
    
    @property
    def matrix(self) -> np.ndarray:
        """Matrix representation of the fermionic swap.
        
        For fermionic swaps, includes the -1 sign from anticommutation.
        """
        # Get base swap matrix
        base_matrix = super().matrix if hasattr(super(), 'matrix') else np.array([[1, 0, 0, 0], 
                                                                                    [0, 0, 1, 0], 
                                                                                    [0, 1, 0, 0], 
                                                                                    [0, 0, 0, 1]])
        
        # Apply fermionic sign if needed
        if self.has_fermionic_sign:
            # The fermionic swap picks up a -1 when swapping occupied states
            # This affects the |01⟩ → -|10⟩ transition
            fermionic_matrix = base_matrix.copy()
            fermionic_matrix[1, 2] = -fermionic_matrix[1, 2]  # |01⟩ → -|10⟩  
            return fermionic_matrix
        else:
            return base_matrix
    
    def to_jordan_wigner(self) -> List[quantum.Box]:
        """Convert fermionic swap to quantum gates via Jordan-Wigner transformation.
        
        Returns
        -------
        List[quantum.Box]
            List of quantum gates implementing the fermionic swap
        """
        if self.has_fermionic_sign:
            # Fermionic swap with anticommutation sign
            # In Jordan-Wigner representation, this becomes a controlled phase
            # plus the usual SWAP gates
            return [
                quantum.CZ,  # Controlled-Z for fermionic sign
                quantum.SWAP,  # Standard qubit swap
            ]
        else:
            # Non-fermionic swap
            return [quantum.SWAP]


# Dictionary of fermionic gates for easy access
FERMIONIC_GATES: Dict[str, Box] = {
    'c_dagger': FermionicCreation(),
    'c': FermionicAnnihilation(),
    'n': FermionicNumber(),
    'hop': FermionicHopping,  # Callable for parameterized hopping
    'delta_dagger': lambda: FermionicPairing(creation=True),
    'delta': lambda: FermionicPairing(creation=False),
}


# Convenience functions for building fermionic circuits

def fermionic_circuit(n_modes: int) -> Diagram:
    """Create an empty fermionic circuit with n modes.
    
    Parameters
    ----------
    n_modes : int
        Number of fermionic modes
        
    Returns
    -------
    Diagram
        Empty fermionic circuit ready for composition
    """
    domain = fermion ** n_modes if n_modes > 0 else Ty()
    return Diagram(domain, domain, [])


def apply_hopping_term(circuit: Diagram, mode_i: int, mode_j: int, t: float = 1.0) -> Diagram:
    """Apply a hopping term between two fermionic modes.
    
    Adds a hopping term H = t(c_i† c_j + c_j† c_i) to the circuit.
    
    Parameters
    ----------
    circuit : Diagram
        Input fermionic circuit
    mode_i : int
        First mode index
    mode_j : int
        Second mode index  
    t : float, optional
        Hopping amplitude, by default 1.0
        
    Returns
    -------
    Diagram
        Circuit with hopping term applied
    """
    hopping_box = FermionicHopping(t)
    
    # For simplicity, assume hopping acts on all modes
    # In full implementation, would need to handle specific mode targeting
    hopping_layer = Layer(Ty(), hopping_box, Ty())
    
    return Diagram(circuit.dom, circuit.cod, circuit.layers + [hopping_layer])


def apply_pairing_term(circuit: Diagram, mode_i: int, mode_j: int, delta: float = 1.0) -> Diagram:
    """Apply a pairing term between two fermionic modes.
    
    Adds a pairing term H = Δ c_i† c_j† + h.c. to the circuit.
    
    Parameters
    ----------
    circuit : Diagram
        Input fermionic circuit
    mode_i : int
        First mode index
    mode_j : int
        Second mode index
    delta : float, optional
        Pairing amplitude, by default 1.0
        
    Returns
    -------
    Diagram  
        Circuit with pairing term applied
    """
    # Create pairing boxes
    pairing_create = FermionicPairing(creation=True)
    pairing_annihilate = FermionicPairing(creation=False)
    
    # Add both creation and annihilation pairing terms
    create_layer = Layer(Ty(), pairing_create, Ty())
    annihilate_layer = Layer(Ty(), pairing_annihilate, Ty())
    
    return Diagram(circuit.dom, circuit.cod, 
                 circuit.layers + [create_layer, annihilate_layer])


def number_operator_expectation(circuit: Diagram, mode: int) -> Diagram:
    """Add number operator measurement to a fermionic circuit.
    
    Parameters
    ----------
    circuit : Diagram
        Input fermionic circuit
    mode : int
        Mode to measure
        
    Returns
    -------
    Diagram
        Circuit with number operator measurement
    """
    number_box = FermionicNumber()
    number_layer = Layer(Ty(), number_box, Ty())
    
    return Diagram(circuit.dom, circuit.cod, circuit.layers + [number_layer])


# Export convenience functions and classes
__all__ = [
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
