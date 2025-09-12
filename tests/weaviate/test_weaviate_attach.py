from attach_tools_from_weaviate import attach_tools_from_query
import json

def test_tool_attachment():
    """Test the Weaviate-based tool search and attachment with various queries"""
    
    test_queries = [
        "find tools for web research and data collection",
        "tools for managing social media content and scheduling posts",
        "tools for data analysis and processing large datasets",
        "tools for natural language processing",
        "tools for connecting to external APIs",
        "tools that can analyze job listings and salary data"
    ]
    
    print("Testing Weaviate Tool Search and Attachment")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("-" * 80)
        
        # Set a higher min_score for more relevant results
        result = attach_tools_from_query(
            query=query,
            target_agent_id="test-agent-id",  # Add required agent ID
            limit=3,
            min_score=75.0
        )
        
        if result["success"]:
            print("\nOperation succeeded!")
            print(f"Message: {result['message']}")
            print("\nSuccessfully attached tools:")
            for tool in result["details"]["successful_attachments"]:
                print(f"\n{tool['name']} (Match: {tool['match_score']:.2f}%)")
                print(f"Tool ID: {tool['tool_id']}")
            
            if result["details"]["failure_count"] > 0:
                print("\nFailed attachments:")
                for failure in result["details"]["failed_attachments"]:
                    print(f"- {failure['name']} ({failure['tool_id']})")
                    print(f"  Error: {failure['error']}")
        else:
            print(f"\nOperation failed: {result['message']}")
            if "details" in result:
                print("\nDetails:")
                print(json.dumps(result["details"], indent=2))
        
        print("\n" + "=" * 80)
        
        # Wait for user input before continuing to next query
        # Comment out for automated testing
        # if i < len(test_queries):
        #     input("\nPress Enter to continue to next query...")

if __name__ == "__main__":
    print("\nThis test will search for relevant tools and attempt to attach them to the default agent.")
    print("Make sure your .env file is properly configured with Weaviate and Letta credentials.")
    # Comment out input for automated testing
    # input("\nPress Enter to begin testing...")
    
    try:
        test_tool_attachment()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {str(e)}")