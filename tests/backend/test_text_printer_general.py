"""Tests for general text printing functionality for all diagram types."""

import pytest

from lambeq.backend.grammar import Box, Cap, Cup, Diagram, Id, Spider, Swap, Ty, Word, Daggered, Frame
from lambeq.backend.quantum import CX, H, qubit, Ket
from lambeq.backend.tensor import Box as TensorBox, Dim
from lambeq.backend.drawing.text_printer import DiagramTextPrinter


class TestGeneralTextPrinter:
    """Test the general text printer for non-pregroup diagrams."""
    
    def test_empty_diagram(self):
        """Test rendering an empty diagram."""
        n = Ty('n')
        diagram = Id(n)
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert result == ""
    
    def test_simple_box_diagram(self):
        """Test rendering a simple box diagram."""
        n = Ty('n')
        s = Ty('s')
        f = Box('f', n, s)
        diagram = Id(n) >> f
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'f' in result
        assert 'n' in result
        assert 's' in result
    
    def test_composite_diagram(self):
        """Test rendering a composite diagram."""
        n = Ty('n')
        s = Ty('s')
        f = Box('f', n, s)
        g = Box('g', s, n)
        diagram = f >> g
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'f' in result
        assert 'g' in result
    
    def test_parallel_composition(self):
        """Test rendering parallel composition."""
        n = Ty('n')
        s = Ty('s')
        f = Box('f', n, s)
        g = Box('g', n, s)
        diagram = f @ g
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'f' in result
        assert 'g' in result
    
    def test_spider_rendering(self):
        """Test rendering spider boxes."""
        n = Ty('n')
        diagram = Word("input", n) >> Spider(n, 1, 3)
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'Spider[1→3]' in result
    
    def test_cap_rendering(self):
        """Test rendering cap boxes."""
        n = Ty('n')
        diagram = Cap(n.r, n).to_diagram()
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        # Check for cap symbol (unicode or ASCII)
        assert '╭' in result or '/' in result
    
    def test_cup_rendering(self):
        """Test rendering cup boxes in non-pregroup context."""
        n = Ty('n')
        diagram = Id(n) @ Id(n.r) >> Cup(n, n.r)
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        # Check for cup symbol
        assert '╰' in result or '\\' in result
    
    def test_swap_rendering(self):
        """Test rendering swap boxes in non-pregroup context."""
        n = Ty('n')
        s = Ty('s')
        diagram = Id(n) @ Id(s) >> Swap(n, s)
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'X' in result
    
    def test_daggered_box(self):
        """Test rendering daggered boxes."""
        n = Ty('n')
        s = Ty('s')
        box = Box('f', n, s)
        daggered = Daggered(box)
        diagram = Id(s) >> daggered
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'f†' in result
    
    def test_frame_box(self):
        """Test rendering frame boxes."""
        n = Ty('n')
        s = Ty('s')
        # Create a simple frame
        inner = Box('inner', n, s)
        frame = Frame('MyFrame', dom=n, cod=s, components=[inner])
        diagram = Id(n) >> frame
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert '[MyFrame]' in result
    
    def test_tensor_diagrams(self):
        """Test rendering tensor diagrams."""
        A = TensorBox("A", Dim(2), Dim(3))
        B = TensorBox("B", Dim(3), Dim(4))
        diagram = A >> B
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'A' in result
        assert 'B' in result
        assert '2' in result
        assert '4' in result
    
    def test_quantum_diagrams(self):
        """Test rendering quantum diagrams."""
        diagram = H @ qubit >> CX
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'H' in result
        assert 'CX' in result
        assert 'qubit' in result
    
    def test_ascii_mode(self):
        """Test ASCII-only rendering mode."""
        n = Ty('n')
        diagram = Cap(n.r, n).to_diagram()
        
        # Test Unicode mode
        printer_unicode = DiagramTextPrinter(use_ascii=False)
        result_unicode = printer_unicode.diagram2str(diagram)
        assert '╭' in result_unicode or '╰' in result_unicode
        
        # Test ASCII mode
        printer_ascii = DiagramTextPrinter(use_ascii=True)
        result_ascii = printer_ascii.diagram2str(diagram)
        assert '/' in result_ascii or '\\' in result_ascii
        assert '╭' not in result_ascii and '╰' not in result_ascii
    
    def test_box_width_handling(self):
        """Test that boxes are wide enough to fit their text."""
        n = Ty('n')
        s = Ty('s')
        long_name = "VeryLongBoxNameThatShouldBeDisplayedProperly"
        box = Box(long_name, n, s)
        diagram = Id(n) >> box
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert long_name in result
    
    def test_no_domain_box(self):
        """Test rendering boxes with no domain."""
        n = Ty('n')
        box = Box('generator', Ty(), n)
        diagram = box >> Id(n)
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'generator' in result
    
    def test_no_codomain_box(self):
        """Test rendering boxes with no codomain."""
        n = Ty('n')
        box = Box('sink', n, Ty())
        diagram = Id(n) >> box
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        assert 'sink' in result
    
    def test_complex_mixed_diagram(self):
        """Test a complex diagram mixing different box types."""
        n = Ty('n')
        s = Ty('s')
        
        # Build a complex diagram
        word1 = Word("start", n)
        box1 = Box('process', n, n @ n)
        box2 = Box('split', n @ n, n @ n @ n)
        box3 = Box('merge', n @ n @ n, s)
        
        diagram = word1 >> box1 >> box2 >> box3
        
        printer = DiagramTextPrinter()
        result = printer.diagram2str(diagram)
        
        # Check all components are present
        assert 'start' in result
        assert 'process' in result
        assert 'split' in result
        assert 'merge' in result
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        printer = DiagramTextPrinter()
        with pytest.raises(ValueError):
            printer.diagram2str("not a diagram")
    
    def test_word_spacing_option(self):
        """Test the word_spacing option."""
        n = Ty('n')
        box1 = Box('A', n, n)
        box2 = Box('B', n, n)
        diagram = box1 @ box2
        
        # Test with different spacing
        printer1 = DiagramTextPrinter(word_spacing=2)
        result1 = printer1.diagram2str(diagram)
        
        printer2 = DiagramTextPrinter(word_spacing=5)
        result2 = printer2.diagram2str(diagram)
        
        # The second result should have more spacing
        assert len(result2) > len(result1)
    
    def test_at_separator_option(self):
        """Test the use_at_separator option."""
        n = Ty('n')
        s = Ty('s')
        box = Box('test', n @ s, n @ s)
        diagram = Id(n @ s) >> box
        
        # Test with dot separator (default)
        printer_dot = DiagramTextPrinter(use_at_separator=False)
        result_dot = printer_dot.diagram2str(diagram)
        assert '·' in result_dot or ' ' in result_dot  # Unicode dot or space
        
        # Test with @ separator
        printer_at = DiagramTextPrinter(use_at_separator=True)
        result_at = printer_at.diagram2str(diagram)
        # Check that we use @ for type separation in the input line
        lines = result_at.split('\n')
        assert any('@' in line for line in lines)


if __name__ == '__main__':
    pytest.main([__file__])
