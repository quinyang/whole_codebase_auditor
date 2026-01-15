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

        # --- OPTIMIZATION: SKIP HUGE FILES EARLY ---
        # Skip json.hpp or lock files before even checking extensions
        if "json.hpp" in item.path or "lock" in item.path:
            print(f"⏭️ Skipping known massive vendor file: {item.path}")
            continue
        
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

            try:
                code_content = code_bytes.decode('utf-8')
            except UnicodeDecodeError:
                print(f"⚠️ Skipping binary/non-utf8 file: {item.path}")
                code_content = "[BINARY DATA SKIPPED]"

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
                "content": code_content,
                "root_type": root.type,
                "statements": root.child_count,
                "has_error": has_error,
                #"tree": tree  
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

def build_mamba_prompt(scanned_files):
    # --- FIX 1: ACTUAL INSTRUCTIONS RESTORED ---
    prompt = """<system_state>
You are the Whole-Codebase Auditor (WCA). 
Your architecture (Mamba SSM) allows you to view this entire repository as a continuous data stream.
Analyze the following files for "Action-at-a-Distance" vulnerabilities, such as:
- Insecure data flows between different modules.
- API keys or secrets hardcoded in one file and exposed in another.
- Logic bugs that only appear when multiple files interact.
</system_state>

<codebase_stream>
"""
    for file in scanned_files:
        # --- FIX 3: SKIP HUGE VENDOR FILES ---
        # Simple heuristic: skip if filename contains "json.hpp" or is larger than 200KB
        if "json.hpp" in file["path"] or file["size"] > 200000:
            print(f"⚠️ Excluding massive file from context: {file['path']}")
            continue

        if "content" not in file or not file["content"]: 
            continue
            
        prompt += f'\n<file path="{file["path"]}">\n'
        prompt += file["content"]
        prompt += f'\n</file>\n'

    prompt += "\n</codebase_stream>"
    
    # --- FIX 2: APPEND THE QUERY ---
    prompt += "\n\nQuery: Identify any cross-file security vulnerabilities in the provided stream."
    
    return prompt


if __name__ == "__main__":
    # Run the scan
    print("🚀 Starting Repo Scan...")
    scanned_results = repo_scan_parser()
    
    print("\n🔗 Stitching Mamba Context...")
    huge_context_string = build_mamba_prompt(scanned_results)
        
    print(f"\n📊 Context Stats:")
    print(f"   Total Length: {len(huge_context_string):,} characters")
    
    # Simple previews
    print("\n" + "="*50)
    print("👀 PREVIEW: HEAD")
    print("="*50)
    print(huge_context_string[:500])  
        
    print("\n" + "="*50)
    print("👀 PREVIEW: TAIL")
    print("="*50)
    print(huge_context_string[-500:])