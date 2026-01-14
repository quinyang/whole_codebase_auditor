import os, importlib
from tree_sitter import Language, Parser
# Import the language modules
import tree_sitter_python as tspython
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo

class LanguageDispatcher:
    def __init__(self):
        # Map extensions to their module and language name
        self.language_map = {
            '.py':   {'package': 'tree_sitter_python',     'name': 'python'},
            '.c':    {'package': 'tree_sitter_c',          'name': 'c'},
            '.h':    {'package': 'tree_sitter_c',          'name': 'c'},
            '.cpp':  {'package': 'tree_sitter_cpp',        'name': 'cpp'},
            '.hpp':  {'package': 'tree_sitter_cpp',        'name': 'cpp'},
            '.cc':   {'package': 'tree_sitter_cpp',        'name': 'cpp'},
            '.cu':   {'package': 'tree_sitter_cpp',        'name': 'cpp'},  # CUDA uses C++ parser
            '.js':   {'package': 'tree_sitter_javascript', 'name': 'javascript'},
            '.go':   {'package': 'tree_sitter_go',         'name': 'go'},
            '.rs':   {'package': 'tree_sitter_rust',       'name': 'rust'},
            '.java': {'package': 'tree_sitter_java',       'name': 'java'},
        }
        
        # Cache parsers so we don't recreate them for every file
        self._parsers = {}
        self._languages = {}
    
    def get_parser_for_file(self, filename):
        """
        Returns a (parser, language) tuple for the given filename.
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()  # Normalize extension to lowercase
        
        # 1. Gracefully handle files with no extension (like .env, .gitignore, Dockerfile)
        if not ext: 
            return None, None
            
        # 2. Skip unsupported extensions silently (or log if needed)
        if ext not in self.language_map:
            return None, None
        
        lang_name = self.language_map[ext]['name']

        # 3. Lazy Load
        if lang_name not in self._parsers:
            try:
                self._load_language(ext)
            except Exception as e:
                print(f"❌ Error loading grammar for {ext}: {e}")
                return None, None

        return self._parsers[lang_name], self._languages[lang_name]
    
    def _load_language(self, ext):
        """
        Internal method to initialize the Language and Parser.
        """
        config = self.language_map[ext]
        pkg_name = config['package']
        
        # Import the module (e.g., import tree_sitter_python)
        language_module = importlib.import_module(pkg_name)
        
        # Get Language Object
        language_obj = Language(language_module.language())
        
        # --- THE FIX IS HERE --- 
        # Old API: parser = Parser(); parser.set_language(language_obj)
        # New API (v0.22+): Parser(language_obj)
        parser = Parser(language_obj)
        
        # Cache
        self._languages[config['name']] = language_obj
        self._parsers[config['name']] = parser
        print(f"✅ Loaded grammar: {config['name'].upper()}")
