import os
import shutil
import re
import datetime
import yaml

# ================= 核心配置区域 (请仔细修改这4项) =================

# 1. 你的 Obsidian 笔记所在的文件夹 (你的源)
# 你当前指向的是一个具体的子主题文件夹
SOURCE_NOTES_PATH = r"C:\Users\23352\Desktop\CS技术栈\CS技术栈\CV\Vision with Deep Learning"

# 2. 你的 Obsidian 图片附件库的【绝对路径】
# 请去你的电脑里找到存放这些 png 的那个文件夹，把完整路径复制到这里！
# 假设它在你的 Vault 根目录下，可能是下面这样（请务必确认）：
OBSIDIAN_IMAGES_PATH = r"C:\Users\23352\Desktop\CS技术栈\CS技术栈\CV\Vision with Deep Learning\attachments" 

# 3. 你希望这批笔记在博客里属于什么分类？
# 格式：["一级分类", "二级分类", "三级分类"...]
# 既然这批笔记是 CV 相关的，我们强制指定分类，不再自动计算
TARGET_CATEGORIES = ["小记", "计算机视觉"]

# 4. Hexo 的 posts 目录 (通常不用改)
HEXO_SOURCE_DIR = "source"
HEXO_POSTS_DIR = os.path.join(HEXO_SOURCE_DIR, "_posts")

# ==============================================================

def setup_directories():
    if not os.path.exists(HEXO_POSTS_DIR):
        os.makedirs(HEXO_POSTS_DIR)

def find_image_file(img_name):
    """
    在指定的图片库绝对路径中查找图片
    """
    # 1. 直接检查根目录
    target_path = os.path.join(OBSIDIAN_IMAGES_PATH, img_name)
    if os.path.exists(target_path):
        return target_path
    
    # 2. 如果图片库里还有子文件夹，递归查找 (防止图片被整理过)
    for root, dirs, files in os.walk(OBSIDIAN_IMAGES_PATH):
        if img_name in files:
            return os.path.join(root, img_name)
    
    return None

def process_file(root, filename):
    file_path = os.path.join(root, filename)
    post_name_no_ext = os.path.splitext(filename)[0]

    # --- 1. 读取内容 ---
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- 2. 处理 Front Matter ---
    front_matter = {}
    body = content
    if content.startswith('---'):
        try:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                front_matter = yaml.safe_load(parts[1]) or {}
                body = parts[2]
        except Exception as e:
            print(f"Warning: Front Matter parse error in {filename}")

    # 补充 Title 和 Date
    if 'title' not in front_matter:
        front_matter['title'] = post_name_no_ext
    if 'date' not in front_matter:
        mtime = os.path.getmtime(file_path)
        front_matter['date'] = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    # 【关键修改】强制使用配置好的分类
    front_matter['categories'] = TARGET_CATEGORIES

    # --- 3. 图片处理与资源文件夹 ---
    # 目标路径: source/_posts/笔记名/
    asset_folder_dir = os.path.join(HEXO_POSTS_DIR, post_name_no_ext)
    
    def replace_image(match):
        img_name = match.group(1) # 获取文件名
        
        # 忽略网络图片
        if img_name.startswith('http') or img_name.startswith('//'):
            return match.group(0)
            
        # 提取纯文件名 (处理 attachment/xxx.png 这种情况)
        img_basename = os.path.basename(img_name)
        
        # 使用绝对路径查找图片
        src_path = find_image_file(img_basename)
        
        if src_path:
            # 创建资源文件夹
            if not os.path.exists(asset_folder_dir):
                os.makedirs(asset_folder_dir)
            
            # 复制图片
            dest_path = os.path.join(asset_folder_dir, img_basename)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                print(f"  [Copy] {img_basename}")
            
            # 返回 Markdown 链接 (hexo-image-link 插件兼容格式)
            return f"![]({img_basename})"
        else:
            print(f"  [Missing] 找不到图片: {img_basename} (请检查 OBSIDIAN_IMAGES_PATH 设置)")
            return match.group(0)

    # 正则替换
    body = re.sub(r'!\[.*?\]\((.*?)\)', replace_image, body)
    
    # --- 4. 写入文件 ---
    new_content = "---\n" + yaml.dump(front_matter, allow_unicode=True, default_flow_style=False) + "---\n" + body
    
    dest_path = os.path.join(HEXO_POSTS_DIR, filename)
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"[Done] {filename} -> {TARGET_CATEGORIES}")

def main():
    setup_directories()
    # 检查图片路径是否存在
    if not os.path.exists(OBSIDIAN_IMAGES_PATH):
        print(f"【错误】图片路径不存在: {OBSIDIAN_IMAGES_PATH}")
        print("请修改脚本中的 OBSIDIAN_IMAGES_PATH 配置！")
        return

    print(f"开始导入... \n源笔记: {SOURCE_NOTES_PATH}\n图片库: {OBSIDIAN_IMAGES_PATH}")
    
    for root, dirs, files in os.walk(SOURCE_NOTES_PATH):
        dirs[:] = [d for d in dirs if not d.startswith('.')] # 忽略隐藏文件夹
        for file in files:
            if file.endswith('.md'):
                process_file(root, file)
    
    print("\n导入结束！建议执行: hexo clean && hexo g && hexo s")

if __name__ == "__main__":
    main()