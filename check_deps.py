import sys
import pkg_resources

required = [
    'flask', 'flask-cors', 'requests', 'fastapi', 'uvicorn', 
    'joblib', 'scikit-learn', 'pandas', 'numpy', 'streamlit', 
    'plotly', 'streamlit-option-menu', 'twilio', 'python-multipart'
]
installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

print(f"Python version: {sys.version}")
print("\nChecking dependencies:")
missing = []
for pkg in required:
    status = installed.get(pkg.lower(), "NOT INSTALLED")
    print(f"{pkg:25}: {status}")
    if status == "NOT INSTALLED":
        missing.append(pkg)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print(f"Run: pip install {' '.join(missing)}")
else:
    print("\nAll required packages are installed.")
