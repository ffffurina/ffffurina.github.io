import os
import shutil
import re
import datetime
import yaml # 需要安装 pyyaml: pip install pyyaml

# ================= 配置区域 =================
# 1. 你的 Obsidian 仓库根目录 (请修改这里!)
# 注意：Windows路径请使用双反斜杠 \\ 或反斜杠 /
OBSIDIAN_VAULT_PATH = r"C:\Users\23352\Desktop\CS技术栈\CS技术栈\CV\Vision with Deep Learning" 

# 2. Obsidian 中存放图片的文件夹名称
ATTACHMENT_FOLDER_NAME = "attachment"

# 3. Hexo 的 source 目录 (通常不用改，除非脚本不在 my-blog 下)
HEXO_SOURCE_DIR = "source"
HEXO_POSTS_DIR = os.path.join(HEXO_SOURCE_DIR, "_posts")
HEXO_IMAGES_DIR = os.path.join(HEXO_SOURCE_DIR, "images")

# 4. 默认顶级分类名称
DEFAULT_CATEGORY = "学习笔记"
# ===========================================

def setup_directories():
    if not os.path.exists(HEXO_POSTS_DIR):
        os.makedirs(HEXO_POSTS_DIR)
    if not os.path.exists(HEXO_IMAGES_DIR):
        os.makedirs(HEXO_IMAGES_DIR)

def process_file(root, filename):
    file_path = os.path.join(root, filename)
    
    # 1. 计算分类 (Categories)
    # 获取相对于 Vault 根目录的路径，例如 "Computer/Language/Python.md"
    rel_path = os.path.relpath(root, OBSIDIAN_VAULT_PATH)
    
    # 如果文件在根目录下，rel_path 是 "."
    if rel_path == ".":
        categories = [DEFAULT_CATEGORY]
    else:
        # 将路径分割为分类列表，例如 ['Computer', 'Language']
        sub_cats = rel_path.split(os.sep)
        # 过滤掉 attachment 文件夹本身（如果笔记不小心放进去了）
        if ATTACHMENT_FOLDER_NAME in sub_cats:
            return
        categories = [DEFAULT_CATEGORY] + sub_cats

    # 2. 读取 Markdown 内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 3. 处理 Front Matter (YAML 头)
    # 检查是否已有 Front Matter
    front_matter = {}
    body = content
    
    if content.startswith('---'):
        try:
            # 简单的 Front Matter 提取
            parts = content.split('---', 2)
            if len(parts) >= 3:
                front_matter = yaml.safe_load(parts[1]) or {}
                body = parts[2]
        except Exception as e:
            print(f"Warning: Failed to parse Front Matter for {filename}: {e}")

    # 补充/覆盖必要的字段
    if 'title' not in front_matter:
        front_matter['title'] = os.path.splitext(filename)[0]
    
    if 'date' not in front_matter:
        # 使用文件修改时间作为日期
        mtime = os.path.getmtime(file_path)
        dt = datetime.datetime.fromtimestamp(mtime)
        front_matter['date'] = dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 合并分类 (保留原有的 tags 等)
    front_matter['categories'] = categories

    # 4. 处理图片链接
    # 匹配模式: ![](filename.png) 或 ![[filename.png]]
    # 这里主要处理标准 Markdown: ![](...)
    
    def replace_image(match):
        img_name = match.group(1)
        # 忽略网络图片 (http 开头)
        if img_name.startswith('http'):
            return match.group(0)
        
        # 尝试在 Obsidian attachment 文件夹中找到该图片
        # 假设图片都在 attachment 文件夹的根目录，或者递归查找
        src_image_path = find_image_in_attachment(img_name)
        
        if src_image_path:
            # 复制图片到 Hexo images 目录
            dest_image_path = os.path.join(HEXO_IMAGES_DIR, img_name)
            if not os.path.exists(dest_image_path):
                shutil.copy2(src_image_path, dest_image_path)
                print(f"  [Image] Copied: {img_name}")
            
            # 返回新的 Hexo 链接格式
            # 使用 basename 确保链接指向 source/images 下的扁平化文件
            return f"![](/images/{os.path.basename(img_name)})"
        else:
            print(f"  [Warning] Image not found: {img_name} in {filename}")
            return match.group(0)

    # 正则替换: ![](path/to/image.png) -> 捕获文件名
    # 简化处理：假设 Obsidian 里引用的是文件名
    # 匹配 ![](...) 里的内容
    body = re.sub(r'!\[.*?\]\((.*?)\)', replace_image, body)

    # 5. 生成新的文件内容
    new_content = "---\n"
    new_content += yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)
    new_content += "---\n"
    new_content += body

    # 6. 写入 Hexo _posts
    # 为了避免文件名冲突，可以保留目录结构，或者直接扁平化（如果有重名文件会覆盖）
    # 这里演示扁平化写入到 _posts，如果需要保留子文件夹也可以
    dest_path = os.path.join(HEXO_POSTS_DIR, filename)
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"[Post] Processed: {filename} -> {categories}")

def find_image_in_attachment(img_name):
    # 在 attachment 文件夹及其子文件夹中搜索图片
    # 简单起见，先直接拼路径，如果你的 attachment 只有一层
    possible_path = os.path.join(OBSIDIAN_VAULT_PATH, ATTACHMENT_FOLDER_NAME, os.path.basename(img_name))
    if os.path.exists(possible_path):
        return possible_path
    
    # 如果 attachment 也有子文件夹，需要遍历查找（会慢一点）
    attach_root = os.path.join(OBSIDIAN_VAULT_PATH, ATTACHMENT_FOLDER_NAME)
    for root, dirs, files in os.walk(attach_root):
        if os.path.basename(img_name) in files:
            return os.path.join(root, os.path.basename(img_name))
    return None

def main():
    setup_directories()
    print(f"Start importing from: {OBSIDIAN_VAULT_PATH}")
    
    for root, dirs, files in os.walk(OBSIDIAN_VAULT_PATH):
        # 忽略隐藏目录和 Hexo 目录（如果重叠）
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != 'public']
        
        for file in files:
            if file.endswith('.md'):
                process_file(root, file)

    print("Import finished!")

if __name__ == "__main__":
    main()