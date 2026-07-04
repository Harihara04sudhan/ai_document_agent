#!/usr/bin/env python3
"""
Setup script for the AI Document Agent.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)


def print_step(step: str):
    """Print a setup step."""
    print(f"\n📌 {step}")
    print("-" * 40)


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def check_python_version():
    """Check if Python version is sufficient."""
    print_step("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def setup_virtual_environment():
    """Set up virtual environment."""
    print_step("Setting up Virtual Environment")
    
    venv_path = Path("ai_document_agent_venv")
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command("python -m venv ai_document_agent_venv", "Virtual environment creation"):
        return False
    
    # Activate and upgrade pip
    if os.name == 'nt':  # Windows
        activate_cmd = ".\\ai_document_agent_venv\\Scripts\\activate"
        pip_cmd = ".\\ai_document_agent_venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source ai_document_agent_venv/bin/activate"
        pip_cmd = "./ai_document_agent_venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Pip upgrade"):
        return False
    
    return True


def install_dependencies():
    """Install required dependencies."""
    print_step("Installing Dependencies")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Determine pip command
    if os.name == 'nt':  # Windows
        pip_cmd = ".\\ai_document_agent_venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        pip_cmd = "./ai_document_agent_venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Dependencies installation"):
        return False
    
    return True


def setup_environment_file():
    """Set up environment configuration."""
    print_step("Setting up Environment Configuration")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy example to .env
    shutil.copy(env_example, env_file)
    print("✅ Created .env file from template")
    
    print("\n⚠️  Important: Please edit .env file and add your Gemini API key:")
    print("   - GEMINI_API_KEY=your_gemini_api_key")
    
    return True


def create_directories():
    """Create necessary directories."""
    print_step("Creating Project Directories")
    
    directories = [
        "documents",
        "data",
        "data/cache",
        "data/vector_db",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    return True


def test_installation():
    """Test the installation."""
    print_step("Testing Installation")
    
    # Determine python command
    if os.name == 'nt':  # Windows
        python_cmd = ".\\ai_document_agent_venv\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        python_cmd = "./ai_document_agent_venv/bin/python"
    
    if not run_command(f"{python_cmd} -c 'import sys; print(\"Python path:\", sys.executable)'", "Python test"):
        return False
    
    # Test imports
    test_imports = [
        "import os",
        "from utils.config import config",
        "print('✅ Configuration loaded successfully')"
    ]
    
    test_code = "; ".join(test_imports)
    if not run_command(f"{python_cmd} -c \"{test_code}\"", "Import test"):
        print("⚠️  Some imports failed, but basic setup is complete")
        return True
    
    return True


def show_next_steps():
    """Show next steps to the user."""
    print_header("Setup Complete! 🎉")
    
    print("📝 Next Steps:")
    print("\n1. Edit the .env file and add your Gemini API key:")
    print("   nano .env")
    print("   # Add: GEMINI_API_KEY=your_actual_api_key")
    
    print("\n2. Add PDF documents to the documents folder:")
    print("   cp /path/to/your/papers/*.pdf documents/")
    
    print("\n3. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   .\\ai_document_agent_venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print("   source ai_document_agent_venv/bin/activate")
    
    print("\n4. Ingest your documents:")
    print("   python ingest_documents.py")
    
    print("\n5. Start using the agent:")
    print("   python main.py --interactive")
    print("   python main.py --query 'your question'")
    print("   python demo.py")
    
    print("\n🔗 Additional Commands:")
    print("   python main.py --health       # Check system health")
    print("   python main.py --stats        # Show document statistics")
    print("   python main.py --arxiv 'query' # Search Arxiv papers")


def main():
    """Main setup function."""
    print_header("AI Document Agent Setup")
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    # Run setup steps
    setup_steps = [
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Environment File", setup_environment_file),
        ("Directories", create_directories),
        ("Installation Test", test_installation)
    ]
    
    for step_name, step_func in setup_steps:
        if not step_func():
            print(f"\n❌ Setup failed at step: {step_name}")
            sys.exit(1)
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
