import os
import json
import pandas as pd

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TSV_FILE = os.path.join(BASE_DIR, "train.tsv")
TEST_TSV_FILE = os.path.join(BASE_DIR, "test.tsv")

JSONL_FILE = os.path.join(BASE_DIR, "mintrec_real.jsonl")
TEST_JSONL_FILE = os.path.join(BASE_DIR, "mintrec_test.jsonl")
IMAGE_DIR = os.path.join(BASE_DIR, "samples", "data")

def convert_dataset(tsv_path, jsonl_path, image_index):
    print(f"正在转换 {os.path.basename(tsv_path)} 到 {os.path.basename(jsonl_path)}...")
    if not os.path.exists(tsv_path):
        print(f"错误: 找不到文件 {tsv_path}")
        return

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"成功读取 TSV，共 {len(df)} 条数据。")

        valid_count = 0
        skipped_count = 0
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                text = str(row['text'])
                original_label = str(row['label'])
                
                # Mapping logic
                aligned_labels = ['Inform', 'Introduce', 'Care', 'Praise', 'Arrange', 'Comfort', 'Greet', 'Agree', 'Apologise', 'Advise', 'Thank', 'Ask for help']
                mild_labels = ['Leave', 'Joke', 'Flaunt']
                severe_labels = ['Oppose', 'Complain', 'Criticize', 'Taunt', 'Prevent']
                
                if original_label in aligned_labels:
                    mapped_label = "aligned"
                elif original_label in mild_labels:
                    mapped_label = "mild"
                elif original_label in severe_labels:
                    mapped_label = "severe"
                else:
                    mapped_label = "mild"
                
                season = str(row['season'])
                episode = str(row['episode'])
                clip = str(row['clip'])
                
                key = f"{season}_{episode}_{clip}"
                
                if key in image_index:
                    image_name = image_index[key]
                    item = {
                        "id": f"mintrec_{index}",
                        "text": text,
                        "image": f"data/{image_name}", 
                        "label": mapped_label,
                        "original_label": original_label 
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    valid_count += 1
                else:
                    skipped_count += 1
                
        print(f"转换完成！成功生成 {valid_count} 条数据到 {jsonl_path}")
        print(f"共跳过 {skipped_count} 条缺失图片的数据。")
        
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")


def convert_mintrec_to_jsonl():
    print(f"开始转换 MintRec 数据集...")
    
    if not os.path.exists(IMAGE_DIR):
        print(f"警告: 找不到图片目录 {IMAGE_DIR}")
        
    print("正在建立图片索引，请稍候...")
    image_index = {}
    if os.path.exists(IMAGE_DIR):
        all_images = os.listdir(IMAGE_DIR)
        for img_name in all_images:
            if not img_name.endswith('.jpg'):
                 continue
            parts = img_name.split('_')
            if len(parts) >= 3:
                key = f"{parts[0]}_{parts[1]}_{parts[2]}" 
                if key not in image_index:
                    image_index[key] = img_name
    
    print(f"索引建立完成，共找到 {len(image_index)} 个唯一视频片段的截图。")
    
    # Process Train
    convert_dataset(TSV_FILE, JSONL_FILE, image_index)
    
    # Process Test
    convert_dataset(TEST_TSV_FILE, TEST_JSONL_FILE, image_index)

if __name__ == "__main__":
    convert_mintrec_to_jsonl()
