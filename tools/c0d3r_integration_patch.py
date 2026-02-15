# Add at top of c0d3r_cli.py imports
    try:
        from enhanced_file_ops import enhanced_ops
        ENHANCED_FILE_OPS = True
    except ImportError:
        ENHANCED_FILE_OPS = False
        print("[WARNING] Enhanced file operations not available")
    pass
    # Replace file search functions with:
    def smart_file_search(filename, context_hint=None):
        if ENHANCED_FILE_OPS:
            result = enhanced_ops.smart_file_search(filename, context_hint)
            print(result["message"])  # Only summary, not full output
            return result["location"]
        else:
            # Fallback to original search logic
            return None
    pass
    # Replace timeout-based commands with:
    def execute_with_fallback(cmd, shell="powershell"):
        if ENHANCED_FILE_OPS:
            result = enhanced_ops.signal_based_command(cmd, shell)
            if not result["success"] and result.get("action") == "restrategize":
                print(f"Command timeout/error: {result['error']} - restrategizing")
            return result
        else:
            # Original subprocess logic
            return None
    