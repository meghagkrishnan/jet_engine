import pandas as pd
import os

def process_file(input_file):
    print(f" {input_file}")
    print(f" {os.getcwd()}")
    
    if not os.path.exists(input_file):
        print(f"erroe {input_file} not exist")
        return

    df = pd.read_csv(input_file, delimiter='\s+', header=None)
    
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df_processed = df.drop(columns=constant_columns)
    

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_processed.csv"
    
    
    df_processed.to_csv(output_file, index=False, header=False)
    print(f" {output_file}")
    


    print(f" {len(df.columns)}")
    print(f" {len(df_processed.columns)}")
    print(f"{len(constant_columns)}")
    print(df_processed.head().to_string(index=False))

input_file = "/Users/zenajo/code/meghagkrishnan/jet_engine/raw_data/test_FD001.txt"
process_file(input_file)