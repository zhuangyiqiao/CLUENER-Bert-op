import json
import re

def is_chinese(text):
    """检查文本是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def validate_label_structure(label):
    """验证 label 的结构"""
    if not isinstance(label, dict):
        return False, "label 不是字典"
    
    for category, entities in label.items():
        if not isinstance(entities, dict):
            return False, f"分类 '{category}' 的值不是字典"
        
        for entity, positions in entities.items():
            if not isinstance(entity, str):
                return False, f"实体 '{entity}' 不是字符串"
            
            if not isinstance(positions, list):
                return False, f"实体 '{entity}' 的位置信息不是列表"
            
            for pos in positions:
                if not isinstance(pos, list) or len(pos) != 2:
                    return False, f"位置信息格式错误: {pos}，应为 [start, end]"
                
                if not all(isinstance(p, int) for p in pos):
                    return False, f"位置信息应为整数，得到: {pos}"
    
    return True, "label 结构正确"

def validate_sample(sample, line_num):
    """验证单个样本"""
    errors = []
    
    # 检查 text
    if "text" not in sample:
        errors.append(f"缺少 'text' 字段")
    elif not isinstance(sample["text"], str):
        errors.append(f"'text' 不是字符串，类型: {type(sample['text'])}")
    elif not is_chinese(sample["text"]):
        errors.append(f"'text' 不包含中文")
    
    # 检查 label
    if "label" not in sample:
        errors.append(f"缺少 'label' 字段")
    else:
        label = sample["label"]
        if label:  # label 不为空时才验证结构
            valid, msg = validate_label_structure(label)
            if not valid:
                errors.append(f"label 结构错误: {msg}")
    
    if errors:
        print(f"\n❌ 第 {line_num} 行错误:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print(f"✓ 第 {line_num} 行正确")
        return True

# 验证前 10 行
with open("data/train.json", "r", encoding="utf-8") as f:
    valid_count = 0
    error_count = 0
    
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            sample = json.loads(line)
            if validate_sample(sample, i):
                valid_count += 1
            else:
                error_count += 1
        except json.JSONDecodeError as e:
            print(f"❌ 第 {i} 行 JSON 解析错误: {e}")
            error_count += 1
        
        if i >= 10749:  # 检查前 10749 行
            break

print(f"\n\n统计: 正确 {valid_count} 行，错误 {error_count} 行")
