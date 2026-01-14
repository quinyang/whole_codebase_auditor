import os, base64, time
from dotenv import load_dotenv
from github import Github, Auth
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from language_dispatcher import LanguageDispatcher

load_dotenv()

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")
TARGET_BRANCH = os.getenv("TARGET_BRANCH", "main")

dispatcher = LanguageDispatcher()

def repo_scan_parser():  

    print("🔹 Authenticate with GitHub...") 
    # Authenticate with GitHub
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        return

    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)

    repo = g.get_repo(GITHUB_REPO)

    # Get from branch 
    print(f"🔹 Fetching file list from branch: {TARGET_BRANCH}...")
    branch_obj = repo.get_branch(TARGET_BRANCH)
    tree = repo.get_git_tree(branch_obj.commit.sha, recursive=True)

    # Find all python files 
    files = [item for item in tree.tree if item.type == "blob"]
    print(f"Found {len(files)} files in the repository.\n")

    results = []

    for index, item in enumerate(files): 
        
        # Skip directories (Type 'tree' = folder, 'blob' = file)
        parser, lang = dispatcher.get_parser_for_file(item.path)

        if parser is None:
            print(f"⏭️ Skipping unsupported file: {item.path}")
            continue  # Skip unsupported file types
        else:
            print(f"🔹 Processing file ({index + 1}/{len(files)}): {item.path}")

        try: 
            # Get Blob
            blob = repo.get_git_blob(item.sha) 

            # Blobs are usually base64 encoded
            if blob.encoding == "base64":
                code_bytes = base64.b64decode(blob.content)
            else:
                code_bytes = blob.content.encode("utf-8")

            # Feed to Tree-sitter (Parser)
            tree = parser.parse(code_bytes)
            root = tree.root_node

            # 2. Check for syntax errors (Optional but recommended for Auditors)
            # If root node has 'ERROR' children, the parser failed to understand some code
            has_error = tree.root_node.has_error

            # Store Result (Metadata for WCA)
            file_data = {
                "path": item.path,
                "size": len(code_bytes),
                "root_type": root.type,
                "statements": root.child_count,
                "has_error": has_error,
                "tree": tree  # We keep the tree object for the next step
            }
            results.append(file_data)

            time.sleep(0.05)  # To avoid hitting rate limits

        except Exception as e:
            print(f"❌ Error accessing repository: {e}")
            return         

    print(f"\n✅ Scan Complete. Processed {len(results)} files.")
    total_bytes = sum(f['size'] for f in results)
    print(f"📊 Total Codebase Size: {total_bytes / 1024:.2f} KB")
    
    return results

if __name__ == "__main__":
    files_endcoding = repo_scan_parser()
    result_ele = files_endcoding[0]
    print("\n📦 Quick Result Element Inspection (1st one)")
    print(f"Key: 'path'              -> {result_ele['path']}")
    print(f"Key: 'size_bytes'        -> {result_ele['size']}")
    print(f"Key: 'root_type'         -> {result_ele['root_type']}")
    print(f"Key: 'child_count'       -> {result_ele['statements']}")
    print(f"Key: 'tree_object'       -> {result_ele['tree']}")