import os

# Define the structure as a dictionary
project_structure = {
    "HoudiniExtensionProject": {
        "CMakeLists.txt": "",
        "LICENSE": "",
        "README.md": "",
        "docs": {
            "API": {},
            "tutorials": {},
            "design_notes.md": ""
        },
        "src": {
            "CMakeLists.txt": "",
            "common": {
                "HoudiniUtils.h": "",
                "HoudiniUtils.cpp": "",
                "CudaUtils.h": "",
                "CudaUtils.cpp": ""
            },
            "cuda_integration": {
                "CMakeLists.txt": "",
                "SOP_CudaIntegration.h": "",
                "SOP_CudaIntegration.cpp": "",
                "kernels": {
                    "advection.cu": "",
                    # Add more kernel files here if needed
                },
                "tests": {
                    "test_advection.cpp": ""
                }
            },
            "vdb_management": {
                "CMakeLists.txt": "",
                "SOP_VDBManagement.h": "",
                "SOP_VDBManagement.cpp": "",
                "tests": {
                    "test_vdb.cpp": ""
                }
            },
            "amr": {
                "CMakeLists.txt": "",
                "SOP_AMR.h": "",
                "SOP_AMR.cpp": "",
                "tests": {
                    "test_amr.cpp": ""
                }
            }
        },
        "tests": {
            "CMakeLists.txt": "",
            "integration_tests.cpp": ""
        },
        "build": {},
        "plugins": {
            "CMakeLists.txt": "",
            "houdini_plugins": {
                "CMakeLists.txt": "",
                "HoudiniPlugin.h": "",
                "HoudiniPlugin.cpp": ""
            },
            "icons": {
                "icon_name.svg": ""
            }
        }
    }
}

def create_structure(base_path, structure):
    """Recursively creates directories and files."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w') as f:
                f.write(content)

# Create the project structure
create_structure(".", project_structure)
