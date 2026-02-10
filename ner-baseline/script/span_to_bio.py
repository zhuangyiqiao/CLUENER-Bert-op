import json  # 用于解析和生成 JSON 数据
import argparse  # 用于解析命令行参数
import os  # 用于文件路径和目录操作
import sys  # 用于访问系统级别的 I/O（打印警告到 stderr）
import glob  # 用于查找 data 目录下的 .json 文件


def span_to_bio(text, label_dict):  # 将 span 格式的标注转换成字符级 BIO 标签
    """Convert span annotations to character-level BIO tags.

    - `text` is a string
    - `label_dict` format: {etype: {span_text: [[start, end], ...], ...}, ...}
    - `start` and `end` are treated as inclusive indices (same as original script)
    """
    tags = ["O"] * len(text)  # 初始化每个字符的标签为 'O'（非实体）

    for etype, entities in label_dict.items():  # 遍历每种实体类型
        for ent, spans in entities.items():  # 遍历该类型下的每个实体文本及其 span 列表
            for start, end in spans:  # 遍历每个 span（start, end）
                # basic validation
                if not isinstance(start, int) or not isinstance(end, int):  # 检查索引是否为整数
                    print(f"[WARN] non-int span for '{ent}' of type {etype}: ({start},{end}) -- skipped", file=sys.stderr)
                    continue  # 非法则跳过该 span
                if start < 0 or end < start or end >= len(text):  # 检查边界有效性（包含端）
                    print(f"[WARN] invalid span for '{ent}' of type {etype}: ({start},{end}) with text length {len(text)} -- skipped", file=sys.stderr)
                    continue  # 越界或反向范围则跳过

                if tags[start] != 'O':  # 若起始位置已经被标注（冲突）
                    print(f"[WARN] overlapping span start at {start} for type {etype} -- existing tag kept", file=sys.stderr)
                    # do not overwrite existing tag at start（保持原标签，不覆盖）
                else:
                    tags[start] = f"B-{etype}"  # 标记起始字符为 B-<类型>

                for i in range(start + 1, end + 1):  # 对后续字符标记为 I-<类型>
                    if tags[i] != 'O':  # 如果该位置已有标签则跳过（避免覆盖冲突）
                        # overlapping internal char; leave existing
                        continue
                    tags[i] = f"I-{etype}"

    return tags  # 返回与文本长度相等的字符级标签列表


def tokens_from_text(text):  # 基于空白的简单分词器（用于 token 级别对齐）
    """Yield (token, start, end_inclusive) using simple whitespace tokenization."""
    for m in re.finditer(r"\S+", text):  # 找到所有非空白连续段
        yield m.group(), m.start(), m.end() - 1  # 返回 token 文本、起始索引、结束（包含端）


def char_tags_to_token_tags(text, char_tags):  # 将字符级标签映射到 token 级标签的简单策略
    """Map character-level BIO tags to token-level tags (simple alignment).

    Rules:
    - If all chars in token are 'O' -> token is 'O'
    - Else find the first non-O char inside token:
      - if that char is at token start and labeled B-<T> -> token label B-<T>
      - else -> token label I-<T>
    - If multiple entity types appear inside a token, a warning is emitted and the first encountered type is used.
    """
    token_labels = []  # 存放 (token, label) 对
    for tok, s, e in tokens_from_text(text):  # 遍历每个 token 及其字符范围
        chunk = char_tags[s:e+1]  # 获取 token 对应字符的标签切片
        if all(t == 'O' for t in chunk):  # 如果全是 'O'
            token_labels.append((tok, 'O'))  # token 标注为 'O'
            continue

        # find first non-O
        first_idx = None  # 第一个非 O 的相对位置
        first_tag = None  # 第一个非 O 的标签
        types_in_chunk = set()  # 记录 token 范围内出现的实体类型集合
        for idx, ct in enumerate(chunk):  # 遍历字符标签切片
            if ct != 'O':  # 找到实体标签
                if first_idx is None:
                    first_idx = idx  # 记录第一个非 O 的索引
                    first_tag = ct  # 记录第一个标签
                if '-' in ct:
                    types_in_chunk.add(ct.split('-', 1)[1])  # 提取实体类型并加入集合
                else:
                    types_in_chunk.add(ct)

        if len(types_in_chunk) > 1:  # 如果 token 内有多种实体类型
            print(f"[WARN] multiple entity types {types_in_chunk} inside token '{tok}' at span ({s},{e}) -- using first found {first_tag}", file=sys.stderr)

        ent_type = first_tag.split('-', 1)[1] if '-' in first_tag else first_tag  # 取第一个标签的实体类型
        if first_idx == 0 and first_tag.startswith('B-'):
            token_label = f"B-{ent_type}"  # 如果第一个非 O 在 token 起始且是 B-，token 用 B-
        else:
            token_label = f"I-{ent_type}"  # 否则用 I-

        token_labels.append((tok, token_label))  # 保存该 token 的标签

    return token_labels  # 返回 token 级别的 (token, label) 列表


def process_file(input_path, output_path, mode='char'):
    """Read a JSONL file and write BIO formatted file.

    Each line of input file should be a JSON object with at least: {"text": ..., "label": ...}
    """
    written = 0  # 统计写入的样本数
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, 1):  # 逐行读取文件，记录行号用于错误报告
            line = line.strip()  # 去除首尾空白
            if not line:  # 跳过空行
                continue
            try:
                sample = json.loads(line)  # 解析 JSON
            except json.JSONDecodeError:
                print(f"[WARN] invalid json on line {lineno} in {input_path} -- skipped", file=sys.stderr)  # JSON 解析失败警告
                continue

            text = sample.get('text', '')  # 获取文本字段，缺省为空字符串
            label = sample.get('label', {})  # 获取标签字段，缺省为空字典
            char_tags = span_to_bio(text, label)  # 生成字符级标签

            # 仅输出字符级 BIO（每行一个字符和对应标签，样本间空行分隔）
            for ch, tg in zip(text, char_tags):  # 逐字符写入 "字符 标签"
                fout.write(f"{ch} {tg}\n")
            fout.write("\n")  # 每个样本之间空行分隔

            written += 1  # 计数

    print(f"Wrote {written} samples from {input_path} -> {output_path}")  # 打印写入结果


def main():  # 命令行入口函数
    p = argparse.ArgumentParser(description='Convert span-labeled JSONL to character-level BIO files.')  # 创建解析器并设置描述
    p.add_argument('input', nargs='*', help='Input JSONL file(s) (one JSON object per line).')  # 输入文件参数（可选，若使用 --all 可不指定）
    p.add_argument('--outdir', default='data', help='Output directory for generated .bio files (default: data)')  # 输出目录参数
    p.add_argument('--all', action='store_true', help='Process all .json files in the data/ directory and generate corresponding .char.bio files. If set, positional inputs are ignored.')
    args = p.parse_args()  # 解析命令行参数

    os.makedirs(args.outdir, exist_ok=True)  # 创建输出目录（如果不存在）

    inputs = []
    if args.all:
        # 优先在项目的 data/ 目录中查找所有 .json 文件，排除预测文件 cluener_predict.json
        data_dir = 'data' if os.path.isdir('data') else os.getcwd()
        all_files = sorted(glob.glob(os.path.join(data_dir, '*.json')))
        # 排除 cluener_predict.json（通常用于模型预测，不用于训练/验证/测试转换）
        inputs = [p for p in all_files if os.path.basename(p) != 'cluener_predict.json']
        if not inputs:
            print(f"[WARN] no .json files found in {data_dir} after excluding cluener_predict.json", file=sys.stderr)
            return
        if len(all_files) != len(inputs):
            print(f"[INFO] excluded cluener_predict.json from processing (--all).", file=sys.stderr)
    else:
        if not args.input:
            print("[ERROR] no input files provided. Use --all to process data/*.json", file=sys.stderr)
            return
        inputs = args.input

    for infile in inputs:  # 处理每个输入文件
        if not os.path.exists(infile):  # 检查输入文件是否存在
            print(f"[ERROR] input file not found: {infile}", file=sys.stderr)
            continue
        base = os.path.splitext(os.path.basename(infile))[0]  # 获取输入文件名（无扩展）
        outname = f"{base}.char.bio"  # 生成输出文件名（固定为字符级）
        outpath = os.path.join(args.outdir, outname)  # 拼接输出路径
        process_file(infile, outpath)  # 调用处理函数（字符级）


if __name__ == '__main__':  # 如果作为脚本运行则执行 main
    main()

