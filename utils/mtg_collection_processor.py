import pandas as pd
import os
from typing import Optional
from pathlib import Path

def process_mtg_collection(
    input_file_path: str,
    output_file_path: Optional[str] = None
) -> bool:
    """
    Process MTG collection CSV file by filtering out rows where Qty equals 0.
    
    Args:
        input_file_path (str): Path to the input CSV file
        output_file_path (Optional[str]): Path for the output CSV file. If None,
            will create a file with '_filtered' suffix in the same directory.
            
    Returns:
        bool: True if processing was successful, False otherwise
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        ValueError: If 'Qty' column is not found in the CSV
    """
    try:
        # Verify input file exists
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
            
        # Read the CSV file
        df = pd.read_csv(input_file_path)
        
        # Verify Qty column exists
        if 'Qty' not in df.columns:
            raise ValueError("Required column 'Qty' not found in the CSV file")
            
        # Filter out rows where Qty = 0
        filtered_df = df[df['Qty'] > 0]
        
        # Generate output path if not provided
        if output_file_path is None:
            input_path = Path(input_file_path)
            output_file_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
            
        # Save filtered data
        filtered_df.to_csv(output_file_path, index=False)
        
        print(f"Successfully processed {len(df)} rows")
        print(f"Filtered out {len(df) - len(filtered_df)} rows with Qty = 0")
        print(f"Saved {len(filtered_df)} rows to: {output_file_path}")
        
        return True
        
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    downloads_path = os.path.expanduser("~/Downloads")
    input_file = os.path.join(downloads_path, "MTG Collection 3-24-25.csv")
    
    # Process the file
    process_mtg_collection(input_file)