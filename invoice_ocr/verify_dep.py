import sys
import importlib.util
import subprocess

def check_package(package_name):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, "Not installed"
        
        # Try to get version
        try:
            mod = importlib.import_module(package_name)
            if hasattr(mod, "__version__"):
                return True, mod.__version__
            elif package_name == "PIL":
                return True, mod.Image.__version__
            return True, "version unknown"
        except:
            return True, "exists but error loading"
    except ImportError:
        return False, "Not installed"

# Check Python packages
packages = ["cv2", "pytesseract", "numpy", "PyPDF2", "pdf2image", "PIL"]
print("Python version:", sys.version)
print("\nChecking Python packages:")
for pkg in packages:
    installed, version = check_package(pkg)
    status = "✅" if installed else "❌"
    print(f"{status} {pkg}: {version}")

# Check Tesseract installation
print("\nChecking Tesseract OCR:")
try:
    result = subprocess.run(["tesseract", "--version"], 
                           capture_output=True, text=True)
    if result.returncode == 0:
        # Extract version from output
        version_line = result.stdout.split('\n')[0]
        version = version_line.split()[1]
        print(f"✅ Tesseract installed: {version}")
        
        # Check languages
        lang_result = subprocess.run(["tesseract", "--list-langs"], 
                                    capture_output=True, text=True)
        langs = [line for line in lang_result.stdout.split('\n') if line]
        print(f"Installed languages: {', '.join(langs[1:])}")
    else:
        print("❌ Tesseract not found or error executing")
except FileNotFoundError:
    print("❌ Tesseract command not found")
except Exception as e:
    print(f"❌ Error checking Tesseract: {str(e)}")

# Check difflib
print("\nChecking difflib:")
try:
    import difflib
    # Create a simple diff
    diff = list(difflib.ndiff("hello".splitlines(), "world".splitlines()))
    print("✅ difflib is available and working")
except Exception as e:
    print(f"❌ difflib error: {str(e)}")
