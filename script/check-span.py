import json

def check_span(sample, line_num):
    """检查单个样本中的 span 是否正确"""
    text = sample.get("text", "")
    labels = sample.get("label", {})
    
    errors = []
    
    if not labels:
        return True, None
    
    for etype, entities in labels.items():
        for ent, positions in entities.items():
            for span in positions:
                if len(span) != 2:
                    errors.append(f"行{line_num}: span 格式错误 {span}")
                    continue
                
                start, end = span
                
                # 检查 span 是否超出 text 长度
                if end > len(text):
                    errors.append(
                        f"行{line_num}: span超出范围 [{start}, {end}]，"
                        f"文本长度{len(text)}，类型:{etype}，实体:{ent}"
                    )
                    continue
                
                # 检查 span 对应的文本是否与实体相符
                # 注意：标注数据中 span 是 [start, end]，但 text[start:end] 是不包括 end 的
                # 所以需要用 text[start:end+1] 来获取完整的实体
                slice_text = text[start:end+1]
                
                if slice_text != ent:
                    errors.append(
                        f"行{line_num}: span不匹配，类型:{etype}，实体:{ent}，"
                        f"span:[{start},{end}]，text[{start}:{end+1}]='{slice_text}'"
                    )
    
    if errors:
        return False, errors
    else:
        return True, None

# 检查所有数据
with open(r"d:\cluener-ner\CLUENER-Bert-op\data\cluener\train.json", "r", encoding="utf-8") as f:
    total = 0
    valid = 0
    error_lines = []
    
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        total += 1
        
        try:
            sample = json.loads(line)
            is_valid, errors = check_span(sample, line_num)
            
            if is_valid:
                valid += 1
            else:
                error_lines.append((line_num, errors))
                
        except json.JSONDecodeError as e:
            error_lines.append((line_num, [f"JSON 解析错误: {e}"]))

print(f"总行数: {total}")
print(f"正确: {valid}")
print(f"错误: {len(error_lines)}\n")

if error_lines:
    print("=" * 80)
    print("错误详情:")
    for line_num, errors in error_lines[:20]:  # 只显示前20个错误
        for error in errors:
            print(error)
    
    if len(error_lines) > 20:
        print(f"\n... 还有 {len(error_lines) - 20} 条错误未显示")
else:
    print("✓ 所有数据都正确！")
