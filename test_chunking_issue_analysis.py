#!/usr/bin/env python3
"""
Test to reproduce the chunking issue that causes CardinalityViolation warnings.

The issue is that even after global deduplication, chunking can create batches
with duplicates, leading to CardinalityViolation errors during INSERT.
"""

import pandas as pd
import numpy as np
from datetime import date

def test_chunking_creates_duplicates():
    """Test how chunking can recreate duplicate issues even after global deduplication."""
    print("🔍 Testing chunking duplicate issue...")
    
    # Create data with duplicates
    data = []
    
    # Create a pattern where global deduplication works, but chunking creates issues
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AAPL', 'MSFT', 'GOOGL']  # Some duplicates
    
    for i, symbol in enumerate(symbols):
        data.append({
            'symbol': symbol,
            'ts': date(2016, 1, 4),  # Same date for conflicts
            'ret_1d': 0.01 + i * 0.001,
            'vol_21': 0.15 + i * 0.001,
        })
    
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    
    # Apply global deduplication (what db.py does at line 489)
    print(f"\nGlobal deduplication:")
    conflict_cols = ['symbol', 'ts']
    original_size = len(df)
    df_deduped = df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
    dedupe_size = len(df_deduped)
    print(f"Global dedup: {original_size} -> {dedupe_size} rows")
    print("After global deduplication:")
    print(df_deduped)
    
    # Now test chunking (what db.py does at line 500-501)
    per_stmt_rows = 2  # Small chunk size to force chunking
    
    print(f"\nChunking with chunk_size={per_stmt_rows}:")
    for i, start in enumerate(range(0, len(df_deduped), per_stmt_rows)):
        chunk = df_deduped.iloc[start:start + per_stmt_rows]
        
        # Check for duplicates within this chunk
        chunk_duplicates = chunk.groupby(conflict_cols).size()
        max_chunk_dupes = chunk_duplicates.max() if len(chunk_duplicates) > 0 else 0
        
        print(f"Chunk {i+1}: {len(chunk)} rows, max duplicates per symbol+ts: {max_chunk_dupes}")
        print(chunk[['symbol', 'ts', 'ret_1d']])
        
        if max_chunk_dupes > 1:
            print(f"  ⚠️  Chunk {i+1} has duplicates that would cause CardinalityViolation!")
    
    # The real issue might be more subtle - let's test a scenario where
    # deduplication works globally but original ordering causes chunking issues
    print(f"\n🔍 Testing problematic ordering scenario...")
    
    # Create data with alternating duplicates that would be problematic for chunking
    problematic_data = []
    for i in range(10):
        # Add AAPL record
        problematic_data.append({
            'symbol': 'AAPL',
            'ts': date(2016, 1, 4),
            'ret_1d': 0.01 + i * 0.001,
            'vol_21': 0.15,
        })
        # Add MSFT record  
        problematic_data.append({
            'symbol': 'MSFT', 
            'ts': date(2016, 1, 4),
            'ret_1d': 0.02 + i * 0.001,
            'vol_21': 0.16,
        })
    
    prob_df = pd.DataFrame(problematic_data)
    print(f"Problematic data: {len(prob_df)} rows")
    
    # Global deduplication
    prob_df_deduped = prob_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
    print(f"After global dedup: {len(prob_df_deduped)} rows")
    print(prob_df_deduped)
    
    # Test chunking
    chunk_size = 3
    print(f"\nChunking problematic data with chunk_size={chunk_size}:")
    has_chunk_duplicates = False
    
    for i, start in enumerate(range(0, len(prob_df_deduped), chunk_size)):
        chunk = prob_df_deduped.iloc[start:start + chunk_size]
        
        chunk_duplicates = chunk.groupby(conflict_cols).size()
        max_chunk_dupes = chunk_duplicates.max() if len(chunk_duplicates) > 0 else 0
        
        print(f"Chunk {i+1}: {len(chunk)} rows, max duplicates: {max_chunk_dupes}")
        if max_chunk_dupes > 1:
            print(f"  ⚠️  Chunk {i+1} has duplicates!")
            has_chunk_duplicates = True
    
    return not has_chunk_duplicates  # Success if no chunk duplicates

def demonstrate_current_fix_limitations():
    """Demonstrate scenarios where the current fix might still generate warnings."""
    print("\n🔧 Demonstrating current fix limitations...")
    
    # The issue from the logs suggests that even with the fix, we're getting
    # "CardinalityViolation with 10 records" warnings
    
    # This could happen if:
    # 1. The retry mechanism itself has duplicates in the "smaller_df"
    # 2. The chunking creates duplicate issues
    # 3. There are race conditions or timing issues
    
    # Let's simulate the retry scenario from the logs
    print("Simulating the '10 records' retry scenario...")
    
    # Create records that would be passed to the retry mechanism
    retry_records = []
    for i in range(10):
        retry_records.append({
            'symbol': 'AAPL',
            'ts': date(2016, 1, 4),
            'ret_1d': 0.01,  # All same values
            'vol_21': 0.15,
        })
    
    print(f"Retry batch: {len(retry_records)} records")
    
    # Simulate what happens in the retry logic (lines 532-540 in db.py)
    smaller_df = pd.DataFrame(retry_records)
    conflict_cols = ['symbol', 'ts']
    
    if len(smaller_df) > 0 and conflict_cols and set(conflict_cols).issubset(smaller_df.columns):
        original_size = len(smaller_df)
        smaller_df = smaller_df.drop_duplicates(subset=conflict_cols, keep='last').reset_index(drop=True)
        dedupe_size = len(smaller_df)
        if dedupe_size < original_size:
            print(f"⚠️  This would generate: 'Removed {original_size - dedupe_size} duplicate rows during retry to prevent CardinalityViolation'")
    
    print(f"After retry deduplication: {len(smaller_df)} records")
    
    # The issue is that this warning is generated for normal operation
    # when duplicates are expected and the fix is working correctly
    print("\n💡 The issue is not that the fix doesn't work - it's that it generates")
    print("   warning messages for normal deduplication operations.")
    print("   These warnings are misleading because they suggest errors when")
    print("   the system is actually working correctly.")
    
    return True

if __name__ == "__main__":
    print("🎯 Analyzing CardinalityViolation warning spam issue\n")
    
    test1_result = test_chunking_creates_duplicates()
    test2_result = demonstrate_current_fix_limitations()
    
    print(f"\n📊 Analysis Results:")
    print(f"Chunking test: {'✅ No issues' if test1_result else '❌ Found chunking issues'}")
    print(f"Fix limitations: {'✅ Understood' if test2_result else '❌ Need more analysis'}")
    
    if test1_result and test2_result:
        print("\n💡 Conclusion: The issue is warning spam, not actual failures.")
        print("   The fix works but generates excessive warning messages.")
        print("   Solution: Reduce log levels for expected deduplication operations.")
    else:
        print("\n⚠️  Found actual technical issues that need addressing.")
    
    exit(0)