# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
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
"""
Text printer
============
Module that allows printing of lambeq pregroup diagrams in text form,
e.g. for the purpose of outputting them graphically in a terminal.

"""

from __future__ import annotations

from dataclasses import dataclass, InitVar
from enum import Enum

from lambeq.backend.grammar import Cup, Diagram, Word, Swap, Spider, Box, Cap, Daggered, Frame


class _MorphismType(Enum):
    """Enumeration for expected morphism types in a diagram."""
    ID = 0
    CUP = 1
    SWAP = 2
    START = -1


@dataclass
class _Morphism:
    """Represents a morphism. `start` and `end` refer to the original
    positions of the involved atomic types in the diagram."""
    morphism: _MorphismType
    start: int
    end: int


UNICODE_CHAR_SET: dict[str, str] = {
    'BAR': '│',
    'TOP_R_CORNER': '╮',
    'TOP_L_CORNER': '╭',
    'BOTTOM_L_CORNER': '╰',
    'BOTTOM_R_CORNER': '╯',
    'LINE': '─',
    'DOT': '·'
}

ASCII_CHAR_SET: dict[str, str] = {
    'BAR': '|',
    'TOP_R_CORNER': chr(160),
    'TOP_L_CORNER': chr(160),
    'BOTTOM_L_CORNER': '\\',
    'BOTTOM_R_CORNER': '/',
    'LINE': '_',
    'DOT': ' '
}


@dataclass
class DiagramTextPrinter:
    """A text printer for all grammar diagrams.

    Parameters
    ----------
    word_spacing : int, default: 2
        The number of spaces between the words of the diagrams.
    use_at_separator : bool, default: False
        Whether to represent types using @ as the monoidal product.
        Otherwise, use the unicode dot character.
    compress_layers : bool, default: True
        Whether to draw boxes in the same layer when they can occur
        simultaneously, otherwise, draw one box per layer.
    use_ascii: bool, default: False
        Whether to draw using ASCII characters only, for
        compatibility reasons.

    """

    word_spacing: int = 2
    use_at_separator: bool = False
    compress_layers: bool = True
    use_ascii: InitVar[bool] = False

    def __post_init__(self, use_ascii: bool) -> None:
        self.chr_set = (UNICODE_CHAR_SET if not use_ascii else ASCII_CHAR_SET)

    def diagram2str(self, diagram: Diagram) -> str:
        """Produces a string that contains a graphical representation of
        the input diagram using text characters.

        Parameters
        ----------
        diagram: :py:class:`lambeq.backend.grammar.Diagram`
            The diagram to be printed.

        Returns
        -------
        str
            String that contains the graphical representation of the
            diagram.
        """
        if not isinstance(diagram, Diagram):
            raise ValueError('The input is not a valid diagram.')
        
        if diagram.is_pregroup:
            # If it's a pregroup diagram, use the specialized printer
            printer = PregroupTextPrinter(self.word_spacing, 
                                         self.use_at_separator, 
                                         self.compress_layers, 
                                         use_ascii=(self.chr_set == ASCII_CHAR_SET))
            return printer.diagram2str(diagram)
        
        # For non-pregroup diagrams, use general rendering
        return self._render_general_diagram(diagram)
    
    def _render_general_diagram(self, diagram: Diagram) -> str:
        """Render a general (non-pregroup) diagram."""
        if not diagram.layers:
            return ""
        
        # Track wire positions throughout the diagram
        # wire_positions maps wire indices to column positions
        wire_positions = {}
        
        # Calculate initial positions for input wires
        type_sep = ' @ ' if self.use_at_separator else self.chr_set['DOT']
        input_types = [str(t) for t in diagram.dom]
        
        # Handle empty domain
        if not input_types:
            current_x = 0
        else:
            # Position input wires with spacing
            current_x = 0
            for i, typ in enumerate(input_types):
                wire_positions[i] = current_x + len(typ) // 2
                current_x += len(typ) + len(type_sep)
        
        # Lists to store all lines to be printed
        lines = []
        
        # Add input types line if there are inputs
        if input_types:
            input_line = type_sep.join(input_types)
            lines.append(input_line)
        
        # Track active wires: maps wire index to its type
        active_wires = {i: diagram.dom[i] for i in range(len(diagram.dom))}
        next_wire_idx = len(diagram.dom)
        
        # Process each layer
        for layer_idx, layer in enumerate(diagram.layers):
            # Render this layer
            layer_lines, wire_positions, active_wires, next_wire_idx = \
                self._render_layer(layer, wire_positions, active_wires, 
                                 next_wire_idx, layer_idx == 0)
            lines.extend(layer_lines)
        
        # Add output types line if there are outputs
        if diagram.cod:
            # Draw final connecting wires
            output_indices = list(sorted(active_wires.keys()))[:len(diagram.cod)]
            if output_indices:
                max_pos = max(wire_positions[idx] for idx in output_indices)
                wire_line = ''
                for idx in output_indices:
                    pos = wire_positions[idx]
                    wire_line = wire_line.ljust(pos) + self.chr_set['BAR']
                lines.append(wire_line)
            
            # Draw output types
            output_line = ''
            for i, idx in enumerate(output_indices):
                if i > 0:
                    output_line += type_sep
                output_line = output_line.ljust(wire_positions[idx] - len(str(diagram.cod[i]))//2)
                output_line += str(diagram.cod[i])
            lines.append(output_line)
        
        return '\n'.join(lines)
    
    def _render_layer(self, layer, wire_positions, active_wires, next_wire_idx, is_first_layer):
        """Render a single layer of the diagram."""
        lines = []
        
        # Get box and offsets
        box = layer.box
        left_wires = layer.left
        right_wires = layer.right
        
        # Calculate box offset (number of wires to the left)
        offset = len(left_wires)
        
        # Get box text representation
        box_text = self._get_box_text(box)
        box_width = len(box_text)
        
        # Get input wire indices for this box
        input_indices = list(sorted(active_wires.keys()))[offset:offset + len(box.dom)]
        
        # Calculate box position
        if input_indices:
            # Position box centered over its input wires
            leftmost = min(wire_positions[idx] for idx in input_indices)
            rightmost = max(wire_positions[idx] for idx in input_indices)
            box_center = (leftmost + rightmost) // 2
        else:
            # No input wires - position after existing wires
            if wire_positions:
                box_center = max(wire_positions.values()) + self.word_spacing + box_width // 2
            else:
                box_center = box_width // 2
        
        box_left = box_center - box_width // 2
        
        # Ensure box doesn't overlap with existing content
        if wire_positions:
            min_left = max(wire_positions.values()) + self.word_spacing
            if box_left < min_left:
                shift = min_left - box_left
                box_left += shift
                box_center += shift
        
        # Draw input wires to box (if not first layer)
        if not is_first_layer and input_indices:
            max_pos = max(wire_positions.values()) if wire_positions else 0
            wire_line = ''
            
            # Draw vertical wires
            all_positions = set()
            for idx in sorted(active_wires.keys()):
                pos = wire_positions[idx]
                all_positions.add(pos)
                if idx in input_indices:
                    # This wire connects to the box
                    wire_line = wire_line.ljust(pos) + self.chr_set['BAR']
                else:
                    # This wire passes through
                    wire_line = wire_line.ljust(pos) + self.chr_set['BAR']
            
            if wire_line.strip():
                lines.append(wire_line)
        
        # Draw the box
        box_line = ' ' * box_left + box_text
        lines.append(box_line)
        
        # Update wire positions and active wires
        new_wire_positions = {}
        new_active_wires = {}
        
        # First, handle wires that don't connect to this box
        for idx in sorted(active_wires.keys()):
            if idx < offset or idx >= offset + len(box.dom):
                # Wire passes through unchanged
                new_wire_positions[idx] = wire_positions[idx]
                new_active_wires[idx] = active_wires[idx]
        
        # Then, add output wires from the box
        if box.cod:
            # Position output wires evenly under the box
            cod_spacing = box_width / (len(box.cod) + 1)
            for i, typ in enumerate(box.cod):
                wire_idx = next_wire_idx + i
                wire_pos = int(box_left + cod_spacing * (i + 1))
                new_wire_positions[wire_idx] = wire_pos
                new_active_wires[wire_idx] = typ
        
        # Handle special cases for Cup and Swap
        if isinstance(box, Cup):
            # Cup consumes two wires, no outputs
            pass
        elif isinstance(box, Swap):
            # Swap exchanges two wires
            if len(input_indices) == 2:
                idx1, idx2 = input_indices
                pos1, pos2 = wire_positions[idx1], wire_positions[idx2]
                # Add swapped wires to new positions
                new_wire_positions[next_wire_idx] = pos2
                new_wire_positions[next_wire_idx + 1] = pos1
                new_active_wires[next_wire_idx] = active_wires[idx2]
                new_active_wires[next_wire_idx + 1] = active_wires[idx1]
        
        # Draw output wires from box
        if box.cod and not isinstance(box, (Cup, Swap)):
            wire_line = ''
            for idx in sorted(new_wire_positions.keys()):
                if idx >= next_wire_idx:
                    # This is a new output wire from the box
                    pos = new_wire_positions[idx]
                    wire_line = wire_line.ljust(pos) + self.chr_set['BAR']
            if wire_line.strip():
                lines.append(wire_line)
        
        # Update next wire index
        if isinstance(box, Cup):
            new_next_idx = next_wire_idx
        elif isinstance(box, Swap):
            new_next_idx = next_wire_idx + 2
        else:
            new_next_idx = next_wire_idx + len(box.cod)
        
        return lines, new_wire_positions, new_active_wires, new_next_idx
    
    def _get_box_text(self, box: Box) -> str:
        """Get the text representation of a box."""
        if isinstance(box, Word):
            return box.name
        elif isinstance(box, Cup):
            # Use Unicode/ASCII cup representation
            if self.chr_set == ASCII_CHAR_SET:
                return '\\_/'
            else:
                return '╰─╯'
        elif isinstance(box, Cap):
            # Use Unicode/ASCII cap representation  
            if self.chr_set == ASCII_CHAR_SET:
                return '/‾\\'
            else:
                return '╭─╮'
        elif isinstance(box, Swap):
            # Use swap symbol
            return 'X'
        elif isinstance(box, Spider):
            # Show spider with its dimensions
            return f'Spider[{len(box.dom)}→{len(box.cod)}]'
        elif isinstance(box, Daggered):
            # Show daggered box
            return f'{box.name}†'
        elif isinstance(box, Frame):
            # Show frame
            return f'[{box.name}]'
        else:
            # Generic box: show name if available, otherwise show type transformation
            if hasattr(box, 'name') and box.name:
                return box.name
            else:
                # Show domain and codomain
                dom_str = 'ε' if not box.dom else str(box.dom)
                cod_str = 'ε' if not box.cod else str(box.cod)
                return f'{dom_str}→{cod_str}'


@dataclass
class PregroupTextPrinter(DiagramTextPrinter):
    """A text printer for pregroup diagrams."""

    def diagram2str(self, diagram: Diagram) -> str:
        """Produces a string that contains a graphical representation of
        the input diagram using text characters. The diagram is expected
        to be in pregroup form, i.e. all words must precede morphisms.

        Parameters
        ----------
        diagram: :py:class:`lambeq.backend.grammar.Diagram`
            The diagram to be printed.

        Returns
        -------
        str
            String that contains the graphical representation of the
            diagram.

        Raises
        ------
        ValueError
            If input is not a pregroup diagram.

        """

        if not (isinstance(diagram, Diagram) and diagram.is_pregroup):
            raise ValueError('The input is not a pregroup diagram.')

        # create headers
        word_sep = ' ' * self.word_spacing
        word_line = ''
        underlines = ''
        type_line = ''
        pos = []
        for box in diagram.boxes:
            if not isinstance(box, Word):
                break

            if word_line:
                word_line += word_sep
                underlines += word_sep
                type_line += word_sep

            word = box.name
            types = [str(ob) for ob in box.cod]
            type_sep = ' @ ' if self.use_at_separator else self.chr_set['DOT']
            type_str = type_sep.join(types)
            width = max(len(word), len(type_str))

            last_pos = len(type_line) + (width - len(type_str)) // 2
            for t in types:
                pos.append(last_pos + (len(t) - 1) // 2)
                last_pos += len(t) + len(type_sep)

            word_line += word.center(width)
            underlines += self.chr_set['LINE'] * width
            type_line += type_str.center(width)

        # process layers
        scan = [*range(len(pos))]
        layers: list[list[_Morphism]] = [[]]
        for box, offset in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, Word):
                continue

            start = scan[offset]
            end = scan[offset + len(box.dom) - 1]
            index = 0
            layer_index = len(layers)
            if self.compress_layers:
                for layer in reversed(layers):
                    conflict = False
                    for i, morphism in enumerate(layer):
                        if morphism.start > end:
                            index = i
                            break
                        elif morphism.end >= start:
                            conflict = True
                            break
                    else:
                        index = len(layer)

                    if conflict:
                        break

                    layer_index -= 1

            morphism = _Morphism(_MorphismType.CUP if isinstance(box, Cup) else
                                 _MorphismType.SWAP, start, end)
            try:
                layers[layer_index].insert(index, morphism)
            except IndexError:
                layers.append([morphism])

            if isinstance(box, Cup):
                del scan[offset:offset + len(box.dom)]

        # draw layers
        print_rows = []
        wires = {i: n for i, n in enumerate(pos)}
        for layer in layers:
            print_rows += self._draw_layer(layer, wires)
            for morphism in layer:
                if morphism.morphism == _MorphismType.CUP:
                    del wires[morphism.start]
                    del wires[morphism.end]

        lines = [word_line.rstrip(), underlines, type_line.rstrip(),
                 *print_rows]
        return '\n'.join(lines)

    def _draw_layer(self,
                    layer: list[_Morphism],
                    wires: dict[int, int]) -> list[str]:
        # `wires` is a mapping from the index of the wire in the input
        # diagram to the location of the wire in the printed output, a
        # column index

        height = 1
        for morphism in layer:
            if morphism.morphism == _MorphismType.SWAP:
                height = 2
                break

        types = {w: _MorphismType.ID for w in wires}
        for morphism in layer:
            types[morphism.start] = _MorphismType.START
            types[morphism.end] = morphism.morphism

        lines = [''] * height
        for idx, t in types.items():
            off = wires[idx]

            if t == _MorphismType.ID:
                for i, line in enumerate(lines):
                    lines[i] = line.ljust(off) + self.chr_set['BAR']
            elif t == _MorphismType.CUP:
                lines[0] += self.chr_set['BOTTOM_L_CORNER']
                lines[0] = (lines[0].ljust(off, self.chr_set['LINE'])
                            + self.chr_set['BOTTOM_R_CORNER'])
            elif t == _MorphismType.SWAP:
                diff = off - len(lines[0])
                lines[1] = (lines[1].ljust(len(lines[0]))
                            + self.chr_set['TOP_L_CORNER']
                            + self.chr_set['BOTTOM_L_CORNER'].center(
                                diff - 1, self.chr_set['LINE'])
                            + self.chr_set['TOP_R_CORNER'])
                lines[0] += (self.chr_set['BOTTOM_L_CORNER']
                             + self.chr_set['TOP_R_CORNER'].center(
                                 diff - 1, self.chr_set['LINE'])
                             + self.chr_set['BOTTOM_R_CORNER'])
            else:
                assert t == _MorphismType.START
                lines[0] = lines[0].ljust(off)

        return lines
