import importlib
import sys

dependencies = {
    "langchain":            "langchain",
    "flask":                "flask",
    "sentence-transformers":"sentence_transformers",
    "pypdf":                "pypdf",
    "python-dotenv":        "dotenv",
    "langchain-pinecone":   "langchain_pinecone",
    "langchain-huggingface":"langchain_huggingface",
    "huggingface-hub":      "huggingface_hub",
    "langchain-community":  "langchain_community",
}

print("=" * 50)
print(f"  Python version: {sys.version.split()[0]}")
print("=" * 50)
print(f"  {'Package':<28} {'Status':<10} Version")
print("=" * 50)

all_ok = True

for package_name, import_name in dependencies.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "N/A")
        print(f"  ✅ {package_name:<26} {'OK':<10} {version}")
    except ImportError as e:
        print(f"  ❌ {package_name:<26} FAILED     {e}")
        all_ok = False

print("=" * 50)
if all_ok:
    print("  ✅ All dependencies are working!")
else:
    print("  ❌ Some dependencies failed. Check above.")
print("=" * 50)