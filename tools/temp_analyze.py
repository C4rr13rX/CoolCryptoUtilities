import re
    with open('c0d3r_cli.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    pass
    # Find imports
    imports = re.findall(r'^(?:from\s+([\w\.]+)\s+)?import\s+([\w\.,\s]+)', content, re.MULTILINE)
    print('=== IMPORTS ===')
    for imp in imports[:20]:
        print(f'{imp[0] or ""} -> {imp[1]}')
    pass
    # Find subprocess/os calls
    subprocess_calls = re.findall(r'subprocess\.(\w+)', content)
    os_calls = re.findall(r'os\.(\w+)', content)
    print('
=== SUBPROCESS CALLS ===')
    print(sorted(set(subprocess_calls)))
    print('
=== OS CALLS ===')
    print(sorted(set(os_calls)))
    pass
    # Find tool-related functions
    tool_funcs = re.findall(r'def\s+(\w*(?:tool|search|run|execute|vm|wallet)\w*)', content, re.IGNORECASE)
    print('
=== TOOL FUNCTIONS ===')
    print(sorted(set(tool_funcs)))
    