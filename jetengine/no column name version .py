import pandas as pd
import os

def process_file(input_file):
    print(f"尝试读取文件: {input_file}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在！")
        return

    # 读取txt文件
    df = pd.read_csv(input_file, delimiter='\s+', header=None)
    
    # 识别并删除所有数值相同的列
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df_processed = df.drop(columns=constant_columns)
    
    # 创建输出文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_processed.csv"
    
    # 将处理后的数据写入新的csv文件
    df_processed.to_csv(output_file, index=False, header=False)
    print(f"已成功创建文件: {output_file}")
    
    # 显示处理后的数据信息
    print("\n处理后的数据信息：")
    print(f"原始列数: {len(df.columns)}")
    print(f"处理后列数: {len(df_processed.columns)}")
    print(f"删除的列数: {len(constant_columns)}")
    print("\n处理后的数据预览（前5行）：")
    print(df_processed.head().to_string(index=False))

# 使用指定的文件路径
input_file = "/Users/zenajo/code/meghagkrishnan/jet_engine/raw_data/test_FD001.txt"
process_file(input_file)