"""
Test file to verify SPARC methodology configuration.
This script checks if the SPARC configuration is properly set up.
"""

import os
import json
import sys


def verify_roomodes_file():
    """Verify the .roomodes file exists and has valid JSON format."""
    roomodes_path = os.path.join(os.getcwd(), '.roomodes')
    
    # Check if file exists
    if not os.path.exists(roomodes_path):
        print("❌ ERROR: .roomodes file not found")
        return False
    
    # Check if content is valid JSON
    try:
        with open(roomodes_path, 'r') as f:
            roomodes_content = json.load(f)
        
        # Verify required modes exist
        required_modes = [
            "sparc", "spec-pseudocode", "architect", "code", 
            "tdd", "debug", "security-review", "docs-writer",
            "integration", "post-deployment-monitoring-mode", 
            "refinement-optimization-mode", "ask", "devops", "tutorial"
        ]
        
        mode_slugs = [mode.get("slug") for mode in roomodes_content.get("customModes", [])]
        missing_modes = [mode for mode in required_modes if mode not in mode_slugs]
        
        if missing_modes:
            print(f"❌ ERROR: Missing required modes: {', '.join(missing_modes)}")
            return False
            
        print("✅ .roomodes file is valid and complete")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: .roomodes file contains invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Failed to validate .roomodes file: {e}")
        return False


def verify_sparc_docs():
    """Verify the SPARC methodology documentation exists."""
    docs_path = os.path.join(os.getcwd(), 'docs', 'SPARC_METHODOLOGY.md')
    
    if not os.path.exists(docs_path):
        print("❌ ERROR: SPARC_METHODOLOGY.md not found in docs directory")
        return False
    
    print("✅ SPARC methodology documentation found")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("🔍 Running SPARC configuration verification tests...")
    
    roomodes_valid = verify_roomodes_file()
    docs_valid = verify_sparc_docs()
    
    if roomodes_valid and docs_valid:
        print("\n🎉 SUCCESS: SPARC methodology setup is complete and valid")
        return True
    else:
        print("\n⚠️ WARNING: SPARC methodology setup has issues that need to be addressed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)