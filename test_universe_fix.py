#!/usr/bin/env python3
"""
Test for the rebuild_universe return type fix.

This test validates that rebuild_universe() returns a list instead of bool,
making it compatible with len() calls in the application.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Set database URL before importing modules
os.environ["DATABASE_URL"] = "sqlite:///test.db"

from data.universe import rebuild_universe


class TestUniverseReturnTypeFix(unittest.TestCase):
    """Test cases for the universe return type fix."""

    def test_empty_symbols_returns_empty_list(self):
        """Test that when no symbols are found, function returns empty list."""
        with patch('data.universe._list_small_cap_symbols') as mock_symbols:
            mock_symbols.return_value = []
            
            result = rebuild_universe()
            
            # Should return empty list, not False
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)
            self.assertEqual(result, [])

    def test_successful_rebuild_returns_symbols_list(self):
        """Test that successful rebuild returns the symbols list."""
        test_symbols = [
            {'symbol': 'TEST1', 'name': 'Test Corp 1'},
            {'symbol': 'TEST2', 'name': 'Test Corp 2'}
        ]
        
        with patch('data.universe._list_small_cap_symbols') as mock_symbols, \
             patch('data.universe.SessionLocal') as mock_session:
            
            mock_symbols.return_value = test_symbols
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            result = rebuild_universe()
            
            # Should return the symbols list
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result, test_symbols)

    def test_database_error_raises_exception(self):
        """Test that database errors are properly re-raised."""
        test_symbols = [{'symbol': 'TEST', 'name': 'Test Corp'}]
        
        with patch('data.universe._list_small_cap_symbols') as mock_symbols, \
             patch('data.universe.SessionLocal') as mock_session:
            
            mock_symbols.return_value = test_symbols
            mock_session.side_effect = Exception("Database connection failed")
            
            # Should raise the exception, not return False
            with self.assertRaises(Exception) as cm:
                rebuild_universe()
            
            self.assertEqual(str(cm.exception), "Database connection failed")

    def test_len_compatibility_with_all_return_cases(self):
        """Test that len() works with all possible return values."""
        # Test empty case
        with patch('data.universe._list_small_cap_symbols') as mock_symbols:
            mock_symbols.return_value = []
            
            result = rebuild_universe()
            # This should not raise TypeError
            size = len(result)
            self.assertEqual(size, 0)

        # Test successful case  
        test_symbols = [{'symbol': 'TEST', 'name': 'Test Corp'}]
        with patch('data.universe._list_small_cap_symbols') as mock_symbols, \
             patch('data.universe.SessionLocal') as mock_session:
            
            mock_symbols.return_value = test_symbols
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            result = rebuild_universe()
            # This should not raise TypeError
            size = len(result)
            self.assertEqual(size, 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)