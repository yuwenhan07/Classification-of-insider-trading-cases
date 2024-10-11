import os
import json
from docx import Document

def docx_to_text(docx_file):
    """
    将.docx文件转化为纯文本。
    :param docx_file: .docx文件路径
    :return: 文档的纯文本内容
    """
    doc = Document(docx_file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def convert_docx_folder_to_json(folder_path, output_json_file):
    """
    将文件夹内所有.docx文件转换为JSON格式，label为文件名，text为文件内容。
    :param folder_path: 包含.docx文件的文件夹路径
    :param output_json_file: 输出的JSON文件路径
    """
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            content = docx_to_text(file_path)
            label = os.path.splitext(filename)[0]  # 去掉扩展名作为label
            data.append({
                'label': label,
                'text': content
            })
    
    # 将数据写入到json文件
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 指定要转换的文件夹路径
    folder_path = '../刑事判决文书'  # 替换为实际文件夹路径
    # 指定输出的JSON文件路径
    output_json_file = 'xs.json'
    
    convert_docx_folder_to_json(folder_path, output_json_file)
    print(f"转换完成，结果保存为 {output_json_file}")