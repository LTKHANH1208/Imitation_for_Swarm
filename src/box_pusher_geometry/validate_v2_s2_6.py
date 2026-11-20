#!/usr/bin/env python3
"""Static checks for the V2-S2.6 additive package."""
from pathlib import Path
import ast
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
errors = []
required = [
    root / 'package.xml', root / 'setup.py', root / 'setup.cfg',
    root / 'box_pusher_geometry' / 'orange_box_geometry_node.py',
    root / 'box_pusher_geometry' / 'geometry.py',
    root / 'config' / 'geometry.yaml',
    root / 'launch' / 'orange_box_geometry.launch.py',
    root / 'test' / 'test_geometry.py',
]
for p in required:
    if not p.exists():
        errors.append(f'MISSING: {p.relative_to(root)}')
for p in root.rglob('*.py'):
    try:
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    except SyntaxError as exc:
        errors.append(f'PYTHON SYNTAX: {p.relative_to(root)}: {exc}')
try:
    ET.parse(root / 'package.xml')
except Exception as exc:
    errors.append(f'PACKAGE XML: {exc}')
node = (root / 'box_pusher_geometry' / 'orange_box_geometry_node.py').read_text(encoding='utf-8')
for forbidden in ('/model/orange_box_', '/box_collection/evaluator/', '/box_collection/ground_truth'):
    # These are checked as runtime subscription defaults; explanatory comments may mention GT.
    if f"create_subscription({forbidden}" in node:
        errors.append(f'GROUND-TRUTH SUBSCRIPTION FOUND: {forbidden}')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print('V2-S2.6 static validation: PASS')
print(f'Python files checked: {sum(1 for _ in root.rglob("*.py"))}')
print('package.xml: PASS')
print('Ground-truth subscriptions: none')
