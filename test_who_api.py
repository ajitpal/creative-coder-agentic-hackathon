#!/usr/bin/env python3
"""
Test script to debug WHO ICD API responses
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.who_icd_client import WHOICDClient

async def test_who_api():
    print("Testing WHO ICD API...")
    client = WHOICDClient()
    
    # Test with hypertension
    result = await client.search_term('hypertension')
    
    print("\n=== WHO ICD API Response ===")
    print(f"Success: {result.get('success')}")
    print(f"Term: {result.get('term')}")
    print(f"Definition: {result.get('definition', 'No definition')}")
    print(f"Results count: {len(result.get('results', []))}")
    
    if result.get('best_match'):
        best_match = result.get('best_match')
        print(f"\nBest match:")
        print(f"  Title: {best_match.get('title', 'No title')}")
        print(f"  Definition: {best_match.get('definition', 'No definition')}")
        print(f"  Code: {best_match.get('code', 'No code')}")
        print(f"  Match score: {best_match.get('match_score', 0)}")
    
    if result.get('results'):
        print(f"\nFirst result:")
        first_result = result['results'][0]
        print(f"  Title: {first_result.get('title', 'No title')}")
        print(f"  Definition: {first_result.get('definition', 'No definition')}")
        print(f"  Code: {first_result.get('code', 'No code')}")
    
    if not result.get('success'):
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_who_api())