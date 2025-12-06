#!/usr/bin/env python3
"""
Test Query Expansion for Multifunctional Tool Discovery

This test verifies that the query expansion module correctly:
1. Detects CRUD operations from natural language queries
2. Identifies domain entities (book, page, issue, agent, etc.)
3. Injects keywords that help find multifunctional tools
4. Maintains backward compatibility with original queries
"""

import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lettaaugment-source'))

import pytest
from query_expansion import (
    QueryExpander,
    OperationType,
    expand_search_query,
    expand_search_query_with_metadata,
    OPERATION_SYNONYMS,
    DOMAIN_KEYWORDS,
    KNOWN_MULTIFUNCTIONAL_TOOLS,
)


class TestOperationDetection:
    """Test detection of CRUD and management operations from queries"""
    
    def setup_method(self):
        self.expander = QueryExpander(enable_aggressive_expansion=True)
    
    def test_detect_create_operation(self):
        """Test that various create synonyms are detected"""
        create_queries = [
            "create a book",
            "make a new page",
            "add an issue",
            "generate a report",
            "build a document",
            "write a chapter",
            "publish a post",
        ]
        for query in create_queries:
            operations = self.expander.detect_operations(query)
            assert OperationType.CREATE in operations, f"Failed to detect CREATE in: {query}"
    
    def test_detect_read_operation(self):
        """Test that various read synonyms are detected"""
        read_queries = [
            "get the book",
            "read the page",
            "fetch agent info",
            "retrieve the document",
            "view the chapter",
            "show me the content",
        ]
        for query in read_queries:
            operations = self.expander.detect_operations(query)
            assert OperationType.READ in operations, f"Failed to detect READ in: {query}"
    
    def test_detect_update_operation(self):
        """Test that various update synonyms are detected"""
        update_queries = [
            "update the book",
            "edit the page",
            "modify the issue",
            "change the content",
            "revise the chapter",
            "fix the document",
        ]
        for query in update_queries:
            operations = self.expander.detect_operations(query)
            assert OperationType.UPDATE in operations, f"Failed to detect UPDATE in: {query}"
    
    def test_detect_delete_operation(self):
        """Test that various delete synonyms are detected"""
        delete_queries = [
            "delete the book",
            "remove the page",
            "erase the issue",
            "destroy the document",
            "clear the content",
        ]
        for query in delete_queries:
            operations = self.expander.detect_operations(query)
            assert OperationType.DELETE in operations, f"Failed to detect DELETE in: {query}"
    
    def test_detect_list_operation(self):
        """Test that list operations are detected"""
        list_queries = [
            "list all books",
            "show all pages",
            "get all issues",
            "browse the library",
            "enumerate chapters",
        ]
        for query in list_queries:
            operations = self.expander.detect_operations(query)
            assert OperationType.LIST in operations, f"Failed to detect LIST in: {query}"


class TestDomainDetection:
    """Test detection of entity domains from queries"""
    
    def setup_method(self):
        self.expander = QueryExpander(enable_aggressive_expansion=True)
    
    def test_detect_book_domain(self):
        """Test book-related domain detection"""
        book_queries = [
            "create a book",
            "find documentation",
            "get the manual",
            "update the guide",
        ]
        for query in book_queries:
            domains = self.expander.detect_domains(query)
            assert "book" in domains, f"Failed to detect 'book' domain in: {query}"
    
    def test_detect_page_domain(self):
        """Test page-related domain detection"""
        page_queries_and_expected = [
            ("create a page", ["page"]),
            ("edit the article", ["article"]),  # article is its own domain
            ("update content", ["content"]),
            ("delete the section", ["page"]),  # section maps to page
        ]
        for query, expected_domains in page_queries_and_expected:
            domains = self.expander.detect_domains(query)
            # Should detect at least one of the expected domains
            assert any(d in domains for d in expected_domains), f"Failed to detect expected domain in: {query}, got: {domains}"
    
    def test_detect_issue_domain(self):
        """Test issue/project management domain detection"""
        # Note: singular forms map to domain, ticket/tickets now included in issue domain
        issue_queries = [
            "create an issue",
            "update the task",
            "delete the bug",
            "list all tickets",  # tickets now maps to issue domain
        ]
        for query in issue_queries:
            domains = self.expander.detect_domains(query)
            assert "issue" in domains, f"Failed to detect 'issue' domain in: {query}, got: {domains}"
    
    def test_detect_agent_domain(self):
        """Test agent-related domain detection"""
        agent_queries = [
            "create an agent",
            "get the assistant",
            "update the bot",
        ]
        for query in agent_queries:
            domains = self.expander.detect_domains(query)
            assert "agent" in domains, f"Failed to detect 'agent' domain in: {query}"


class TestQueryExpansion:
    """Test the full query expansion process"""
    
    def setup_method(self):
        self.expander = QueryExpander(enable_aggressive_expansion=True)
    
    def test_expansion_adds_crud_keywords(self):
        """Test that CRUD operations add relevant keywords"""
        result = self.expander.expand_query("create a book")
        
        # Should include CRUD-related keywords
        assert "crud" in result.expanded_query.lower()
        assert result.confidence > 0.5
    
    def test_expansion_preserves_original(self):
        """Test that original query is preserved"""
        original = "create a book in bookstack"
        result = self.expander.expand_query(original)
        
        # Original should be at the start
        assert result.expanded_query.startswith(original)
        assert result.original_query == original
    
    def test_expansion_for_bookstack_crud(self):
        """Test expansion specifically for BookStack CRUD discovery"""
        result = self.expander.expand_query("create a book")
        
        # Should add keywords that help find bookstack_content_crud
        expanded_lower = result.expanded_query.lower()
        
        # Check for operation keywords
        assert any(kw in expanded_lower for kw in ["create", "make", "add", "new"])
        
        # Check for domain keywords
        assert any(kw in expanded_lower for kw in ["book", "document", "documentation"])
        
        # Check for multifunctional tool keywords
        assert "crud" in expanded_lower or "manage" in expanded_lower
    
    def test_expansion_metadata(self):
        """Test that expansion returns proper metadata"""
        result = expand_search_query_with_metadata("update the page")
        
        assert result.original_query == "update the page"
        assert len(result.added_keywords) > 0
        assert OperationType.UPDATE in result.detected_operations
        assert "page" in result.detected_domains
    
    def test_no_duplicate_keywords(self):
        """Test that keywords aren't duplicated"""
        result = self.expander.expand_query("create a book")
        words = result.expanded_query.lower().split()
        
        # Count occurrences (some might legitimately appear twice from different sources)
        # but "create" and "book" from original shouldn't be in added_keywords
        assert "create" not in result.added_keywords
        assert "book" not in result.added_keywords


class TestMultifunctionalToolMapping:
    """Test that known multifunctional tools are properly mapped"""
    
    def test_bookstack_crud_mapping(self):
        """Verify BookStack CRUD tool mapping is correct"""
        assert "bookstack_content_crud" in KNOWN_MULTIFUNCTIONAL_TOOLS
        
        tool_info = KNOWN_MULTIFUNCTIONAL_TOOLS["bookstack_content_crud"]
        assert "create" in tool_info["operations"]
        assert "book" in tool_info["domains"]
        assert "bookstack" in tool_info["keywords"]
    
    def test_letta_agent_mapping(self):
        """Verify Letta agent tool mapping is correct"""
        assert "letta_agent_advanced" in KNOWN_MULTIFUNCTIONAL_TOOLS
        
        tool_info = KNOWN_MULTIFUNCTIONAL_TOOLS["letta_agent_advanced"]
        assert "create" in tool_info["operations"]
        assert "agent" in tool_info["domains"]
    
    def test_expansion_matches_known_tools(self):
        """Test that expansion adds keywords matching known tools"""
        expander = QueryExpander(enable_aggressive_expansion=True)
        
        # Query that should match bookstack_content_crud
        result = expander.expand_query("create a book")
        
        # Should include some BookStack-specific keywords
        tool_keywords = KNOWN_MULTIFUNCTIONAL_TOOLS["bookstack_content_crud"]["keywords"]
        expanded_lower = result.expanded_query.lower()
        
        # At least some tool-specific keywords should be present
        matches = [kw for kw in tool_keywords if kw.lower() in expanded_lower]
        assert len(matches) > 0, f"Expected some tool keywords to be added. Got: {expanded_lower}"


class TestConvenienceFunctions:
    """Test the module-level convenience functions"""
    
    def test_expand_search_query(self):
        """Test the simple expand function"""
        original = "delete the page"
        expanded = expand_search_query(original)
        
        assert len(expanded) > len(original)
        assert original in expanded
    
    def test_expand_with_metadata(self):
        """Test the detailed expand function"""
        result = expand_search_query_with_metadata("list all agents")
        
        assert hasattr(result, 'original_query')
        assert hasattr(result, 'expanded_query')
        assert hasattr(result, 'detected_operations')
        assert hasattr(result, 'detected_domains')
        assert hasattr(result, 'added_keywords')
        assert hasattr(result, 'confidence')


class TestEdgeCases:
    """Test edge cases and unusual inputs"""
    
    def setup_method(self):
        self.expander = QueryExpander(enable_aggressive_expansion=True)
    
    def test_empty_query(self):
        """Test handling of empty query"""
        result = self.expander.expand_query("")
        assert result.expanded_query == ""
        assert result.confidence == 0.5  # No operations or domains detected
    
    def test_query_with_no_operations(self):
        """Test query that doesn't match any operations"""
        result = self.expander.expand_query("hello world")
        
        assert OperationType.UNKNOWN in result.detected_operations
        assert result.confidence < 1.0
    
    def test_query_with_multiple_operations(self):
        """Test query mentioning multiple operations"""
        result = self.expander.expand_query("create and delete books")
        
        operations = result.detected_operations
        assert OperationType.CREATE in operations
        assert OperationType.DELETE in operations
    
    def test_case_insensitivity(self):
        """Test that detection is case insensitive"""
        result1 = self.expander.expand_query("CREATE a BOOK")
        result2 = self.expander.expand_query("create a book")
        
        # Should detect same operations regardless of case
        assert result1.detected_operations == result2.detected_operations
        assert result1.detected_domains == result2.detected_domains


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
