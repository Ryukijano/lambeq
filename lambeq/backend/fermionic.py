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
from typing import Any, Dict, cast, Optional, Union
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
        
        # For now, delegate to quantum eval method
        # TODO: Implement efficient fermionic evaluation using covariance matrices
        return super().eval(*others, 
                          backend=backend,
                          mixed=mixed,
                          contractor=contractor or tn.contractors.auto,
                          **params)
    
    def to_tn(self, mixed=False):
        """Convert to tensor network representation.
        
        For fermionic circuits, this creates a tensor network that
        respects the fermionic anticommutation relations.
        
        Parameters
        ----------
        mixed : bool, optional
            Whether to create mixed state representation, by default False
            
        Returns
        -------
        tuple
            Nodes and output edge order for the tensor network
        """
        
        # For now, delegate to quantum implementation
        # TODO: Implement fermionic-specific tensor network conversion
        return super().to_tn(mixed=mixed)
    
    def to_tk(self):
        """Convert to tket circuit representation.
        
        Converts the fermionic diagram to a tket circuit, handling
        the Jordan-Wigner transformation appropriately.
        
        Returns
        -------
        Circuit
            A tket circuit representation
        """
        
        # For now, delegate to quantum implementation  
        # TODO: Implement fermionic-specific tket conversion
        return super().to_tk()
    
    def to_pennylane(self, probabilities=False, backend_config=None,
                    diff_method='best', **kwargs):
        """Convert to PennyLane circuit representation.
        
        Converts the fermionic diagram to a PennyLane circuit with
        appropriate handling of fermionic operations.
        
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
        
        # For now, delegate to quantum implementation
        # TODO: Implement fermionic-specific PennyLane conversion
        return super().to_pennylane(probabilities=probabilities,
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


# Fermionic special boxes - these will need to be implemented specifically
# for fermionic circuits to maintain proper anticommutation relations

@Diagram.register_special_box('cap')
def generate_cap(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a fermionic cap diagram.
    
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
    
    # For now, delegate to quantum implementation
    # TODO: Implement fermionic-specific cap with proper anticommutation
    return cast(Diagram, quantum.generate_cap(left, right, is_reversed))


@Diagram.register_special_box('cup')
def generate_cup(left: Ty, right: Ty, is_reversed=False) -> Diagram:
    """Generate a fermionic cup diagram.
    
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
    
    # For now, delegate to quantum implementation
    # TODO: Implement fermionic-specific cup with proper anticommutation
    return cast(Diagram, quantum.generate_cup(left, right, is_reversed))


@Diagram.register_special_box('spider')
def generate_spider(type: Ty, n_legs_in: int, n_legs_out: int) -> Diagram:
    """Generate a fermionic spider diagram.
    
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
    
    # For now, delegate to quantum implementation
    # TODO: Implement fermionic-specific spider with proper anticommutation
    return cast(Diagram, quantum.generate_spider(type, n_legs_in, n_legs_out))


@Diagram.register_special_box('swap')
class Swap(quantum.Swap, Box):
    """A fermionic swap box.
    
    In fermionic systems, swaps introduce a minus sign due to
    anticommutation relations of fermions: ψ_i ψ_j = -ψ_j ψ_i
    """
    
    def __init__(self, left: Ty, right: Ty):
        super().__init__(left, right)
        # In a full implementation, this would include fermionic sign logic
        # For now, we note the difference in the docstring
        
    @property
    def has_fermionic_sign(self) -> bool:
        """Whether this swap introduces a fermionic sign.
        
        Returns
        -------
        bool
            True if this swap involves fermionic modes and thus 
            requires a -1 phase factor.
        """
        return self.left == fermion and self.right == fermion


# Dictionary of fermionic gates for easy access
FERMIONIC_GATES: Dict[str, Box] = {
    'c_dagger': FermionicCreation(),
    'c': FermionicAnnihilation(),
    'n': FermionicNumber(),
    'hop': FermionicHopping,  # Callable for parameterized hopping
    'delta_dagger': lambda: FermionicPairing(creation=True),
    'delta': lambda: FermionicPairing(creation=False),
}
