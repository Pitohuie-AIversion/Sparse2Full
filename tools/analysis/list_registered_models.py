import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.registry import list_models
# Trigger registration
import models.spatial 
import models.temporal

print("Registered Models:")
for m in list_models():
    print(m)
