# ASCII Drawing Implementation for All Lambeq Diagrams

## Summary

Successfully implemented ASCII drawing functionality for all lambeq diagram types (grammar, tensor, and quantum), extending beyond the previous limitation of only supporting pregroup diagrams.

## Changes Made

### 1. Extended `DiagramTextPrinter` Class
- Implemented the `diagram2str` method in the base `DiagramTextPrinter` class
- Added logic to detect pregroup diagrams and delegate to the specialized printer
- Created `_render_general_diagram` method for handling arbitrary diagrams
- Implemented `_render_layer` method for processing individual diagram layers
- Added `_get_box_text` method for generating text representations of different box types

### 2. Updated Drawing Module
- Modified `render_as_str` function in `drawing.py` to use the general `DiagramTextPrinter`
- Removed the `NotImplementedError` for non-pregroup diagrams

### 3. Box Type Support
The implementation now supports rendering of:
- **Word** boxes: Display the word name
- **Cup** boxes: Unicode `╰─╯` or ASCII `\_/`
- **Cap** boxes: Unicode `╭─╮` or ASCII `/‾\`
- **Swap** boxes: Display as `X`
- **Spider** boxes: Show as `Spider[n→m]` with input/output counts
- **Daggered** boxes: Display with `†` symbol
- **Frame** boxes: Show as `[name]`
- **Generic** boxes: Display name or show transformation `dom→cod`

### 4. Features
- **Box width calculation**: Automatically adjusts box width to fit text content
- **Wire tracking**: Maintains wire positions throughout the diagram
- **Type display**: Shows input and output types with configurable separators
- **ASCII mode**: Supports both Unicode and ASCII-only rendering
- **Configurable spacing**: Adjustable word spacing between boxes
- **Type separator**: Option to use `@` or `·` for type composition

### 5. Test Coverage
Created comprehensive unit tests in `tests/backend/test_text_printer_general.py` covering:
- Empty diagrams
- Simple and composite diagrams
- Parallel composition
- All special box types (Spider, Cap, Cup, Swap, Daggered, Frame)
- Tensor and quantum diagrams
- ASCII vs Unicode modes
- Configuration options
- Edge cases (no domain/codomain boxes)

## Usage Examples

```python
from lambeq.backend.grammar import Box, Ty, Word, Cup, Id

# Simple grammar diagram
n = Ty('n')
s = Ty('s')
diagram = Word("Alice", n) @ Word("runs", n.r @ s) >> Cup(n, n.r) @ Id(s)
print(diagram.render_as_str())

# Output:
# Alice   runs
# ─────  ─────
#   n    n.r·s
#   ╰─────╯  │

# Non-pregroup diagram with generic boxes
f = Box('process', n, s)
g = Box('analyze', s, n)
diagram = f >> g
print(diagram.render_as_str())

# Tensor diagram
from lambeq.backend.tensor import Box as TensorBox, Dim
A = TensorBox("A", Dim(2), Dim(3))
B = TensorBox("B", Dim(3), Dim(4))
diagram = A >> B
print(diagram.render_as_str())

# Quantum circuit
from lambeq.backend.quantum import H, CX, qubit
diagram = H @ qubit >> CX
print(diagram.render_as_str())
```

## Testing
All tests pass successfully:
- 20/20 tests pass in `test_text_printer_general.py`
- 8/8 tests pass in existing `test_text_printer.py`

The implementation maintains backward compatibility with existing pregroup diagram rendering while adding support for all other diagram types in lambeq.
