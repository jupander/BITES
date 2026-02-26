# inject_comments_v9.py
# Refactored: Data-driven approach - all data is preserved,
# template only provides formatting (comments, blank lines, key order)

import yaml
import re
import sys
from collections import OrderedDict

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def warn(message, details=None) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ WARNING:{Colors.RESET} {Colors.YELLOW}{message}{Colors.RESET}", file=sys.stderr)
    if details:
        for detail in details:
            print(f"  {Colors.YELLOW}→ {detail}{Colors.RESET}", file=sys.stderr)


def error(message, details=None) -> None:
    """Print an error message."""
    print(f"{Colors.RED}{Colors.BOLD}✖ ERROR:{Colors.RESET} {Colors.RED}{message}{Colors.RESET}", file=sys.stderr)
    if details:
        for detail in details:
            print(f"  {Colors.RED}→ {detail}{Colors.RESET}", file=sys.stderr)


def info(message) -> None:
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.RESET}", file=sys.stderr)


def success(message) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}{Colors.BOLD}✔ {message}{Colors.RESET}", file=sys.stderr)


def load_data(path) -> dict:
    """Load YAML data file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def extract_template_formatting(template_path) -> tuple:
    """
    Extract formatting metadata from template:
    - inline_comments: dict mapping dotted key paths to inline comments
    - preceding_content: dict mapping dotted key paths to preceding blank lines/comments
    - key_order: list of dotted key paths in template order
    
    Template values are ignored - only formatting is extracted.
    """
    with open(template_path, 'r') as f:
        lines = f.readlines()

    inline_comments = {}
    preceding_content = {}
    key_order = []

    indent_stack = []  # [(indent_level, key_name), ...]
    pending_content = []  # blank lines and comment lines before next key

    for line in lines:
        stripped = line.strip()

        # Blank line or comment-only line -> accumulate as pending content
        if not stripped or stripped.startswith('#'):
            pending_content.append(line)
            continue

        # Try to parse a key line: "  key_name: value  # comment"
        match = re.match(r'^(\s*)([\w_-]+):\s*(?:.*?)(\s*#.*)?$', line)
        if match:
            indent_str, key, comment = match.groups()
            indent_level = len(indent_str)

            # Pop stack to find correct parent based on indentation
            while indent_stack and indent_stack[-1][0] >= indent_level:
                indent_stack.pop()

            # Build full dotted path
            if indent_stack:
                parent_path = '.'.join(k for _, k in indent_stack)
                full_path = f"{parent_path}.{key}"
            else:
                full_path = key

            # Store formatting metadata
            key_order.append(full_path)
            if comment:
                inline_comments[full_path] = comment.rstrip()
            if pending_content:
                preceding_content[full_path] = pending_content.copy()
                pending_content.clear()

            # Push to stack for potential nested keys
            indent_stack.append((indent_level, key))

        elif re.match(r'^\s*-\s', line):
            # List item - skip (we'll regenerate from data)
            continue
        else:
            # Unknown line format - preserve as pending content
            pending_content.append(line)

    # Capture any trailing comments/blank lines
    trailing_content = pending_content if pending_content else None

    return inline_comments, preceding_content, key_order, trailing_content


def flatten_data(d, prefix='') -> OrderedDict:
    """
    Flatten nested dict into OrderedDict with dot-separated keys.
    Dicts get a None value (section header), leaves get their actual value.
    """
    items = OrderedDict()
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items[full_key] = None  # Section header marker
            items.update(flatten_data(v, full_key))
        else:
            items[full_key] = v
    return items


def format_yaml_value(value, indent_str="") -> str:
    """Format a value for YAML output."""
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, list):
        if not value:
            return "[]"
        # Block-style list
        list_lines = []
        for item in value:
            formatted_item = format_yaml_value(item, indent_str + "  ")
            # Handle nested structures in lists
            if isinstance(item, dict):
                # For dict items, format as inline or block
                dict_str = yaml.safe_dump(item, default_flow_style=True).strip()
                list_lines.append(f"{indent_str}  - {dict_str}")
            else:
                list_lines.append(f"{indent_str}  - {formatted_item}")
        return "\n" + "\n".join(list_lines)
    elif isinstance(value, str):
        # Quote strings that contain special YAML characters
        if re.search(r'[:#\[\]{}&*!|>\'\"%@`]', value) or \
           value.lower() in ('true', 'false', 'null', 'yes', 'no', 'on', 'off') or \
           value == '' or \
           re.match(r'^[\d.]+$', value):
            # Use double quotes and escape internal quotes
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return value
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return str(value)


def merge_key_orders(template_order, data_keys) -> list:
    """
    Merge template key order with data keys.
    - Keys in both: use template order
    - Keys only in data: insert after their parent (or at end of parent's section)
    - Keys only in template: ignored (not in data)
    """
    result = []
    data_keys_set = set(data_keys)
    added = set()

    for tmpl_key in template_order:
        if tmpl_key in data_keys_set and tmpl_key not in added:
            result.append(tmpl_key)
            added.add(tmpl_key)

            # Check if there are data keys that are children of this key
            # but weren't in the template - add them right after
            for data_key in data_keys:
                if data_key not in added and data_key.startswith(tmpl_key + '.'):
                    # Check it's a direct child or nested child not covered by template
                    result.append(data_key)
                    added.add(data_key)

    # Add any remaining data keys not in template (preserve their original order)
    for data_key in data_keys:
        if data_key not in added:
            # Try to find best insertion point (after parent)
            parts = data_key.split('.')
            inserted = False
            for i in range(len(parts) - 1, 0, -1):
                parent = '.'.join(parts[:i])
                if parent in added:
                    # Find position after parent and its children
                    try:
                        parent_idx = result.index(parent)
                        insert_idx = parent_idx + 1
                        while insert_idx < len(result) and result[insert_idx].startswith(parent + '.'):
                            insert_idx += 1
                        result.insert(insert_idx, data_key)
                        inserted = True
                        break
                    except ValueError:
                        pass
            if not inserted:
                result.append(data_key)
            added.add(data_key)

    return result


def inject_into_template(template_path, data_path, output_path) -> bool:
    """
    Main function: inject data into template format.
    
    - ALL data from data_path is preserved in output
    - Template only provides: comments, blank lines, key ordering
    - Template values are completely ignored
    """
    print(f"\n{'='*60}", file=sys.stderr)
    info(f"Processing: {data_path} + {template_path} → {output_path}")
    print(f"{'='*60}\n", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Load and validate data file
    # -------------------------------------------------------------------------
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        error(f"Data file not found: {data_path}")
        return False
    except yaml.YAMLError as e:
        error(f"Failed to parse data file: {data_path}", [str(e)])
        return False

    if data is None:
        error(f"Data file is empty or contains only null: {data_path}")
        return False

    if not isinstance(data, dict):
        error(f"Data file root must be a dictionary, got: {type(data).__name__}")
        return False

    flat_data = flatten_data(data)
    data_keys = set(flat_data.keys())

    # -------------------------------------------------------------------------
    # Load and validate template file
    # -------------------------------------------------------------------------
    try:
        inline_comments, preceding_content, template_order, trailing = \
            extract_template_formatting(template_path)
    except FileNotFoundError:
        error(f"Template file not found: {template_path}")
        return False
    except Exception as e:
        error(f"Failed to parse template file: {template_path}", [str(e)])
        return False

    template_keys = set(template_order)

    if not template_keys:
        warn("Template file contains no keys - output will have no formatting from template")

    # -------------------------------------------------------------------------
    # Analyze key mismatches
    # -------------------------------------------------------------------------
    keys_only_in_data = data_keys - template_keys
    keys_only_in_template = template_keys - data_keys
    keys_in_both = data_keys & template_keys

    # Report: Keys in data but not in template (will be added without formatting)
    if keys_only_in_data:
        # Separate into section headers and leaf keys
        section_headers = [k for k in keys_only_in_data if flat_data.get(k) is None]
        leaf_keys = [k for k in keys_only_in_data if flat_data.get(k) is not None]
        
        if section_headers:
            warn(
                f"Data contains {len(section_headers)} section(s) NOT in template (will be added without comments):",
                sorted(section_headers)
            )
        if leaf_keys:
            warn(
                f"Data contains {len(leaf_keys)} key(s) NOT in template (will be added without comments):",
                sorted(leaf_keys)
            )

    # Report: Keys in template but not in data (formatting will be lost)
    if keys_only_in_template:
        warn(
            f"Template contains {len(keys_only_in_template)} key(s) NOT in data (formatting will be lost):",
            sorted(keys_only_in_template)
        )
        # Check for lost comments
        lost_comments = [k for k in keys_only_in_template if k in inline_comments]
        if lost_comments:
            warn(
                f"The following template comments will be LOST (keys not in data):",
                [f"{k}: {inline_comments[k]}" for k in sorted(lost_comments)]
            )

    # Report: Check for type mismatches (template has leaf, data has section or vice versa)
    type_mismatches = []
    for key in keys_in_both:
        # We can't easily know template types, but we can detect if data has nested dict
        # where template might expect a leaf
        pass  # Future enhancement: detect type mismatches

    # -------------------------------------------------------------------------
    # Check for potential ordering issues
    # -------------------------------------------------------------------------
    if template_order:
        data_key_list = list(flat_data.keys())
        template_filtered = [k for k in template_order if k in data_keys]
        data_filtered = [k for k in data_key_list if k in template_keys]
        
        if template_filtered != data_filtered:
            warn(
                "Key ordering differs between template and data",
                [
                    "Output will follow template order for matching keys",
                    "Data-only keys will be inserted after their parent section"
                ]
            )

    # -------------------------------------------------------------------------
    # Merge key orders and generate output
    # -------------------------------------------------------------------------
    final_key_order = merge_key_orders(template_order, list(flat_data.keys()))

    output_lines = []
    keys_without_comments = []
    keys_with_comments = []

    for key in final_key_order:
        if key not in flat_data:
            continue  # Skip template-only keys (not in data)

        parts = key.split('.')
        depth = len(parts) - 1
        indent_str = "  " * depth
        leaf_name = parts[-1]
        value = flat_data[key]

        # Add preceding content (blank lines, comments) from template
        if key in preceding_content:
            output_lines.extend(preceding_content[key])

        # Get inline comment from template (if any)
        comment = inline_comments.get(key, "")
        
        # Track comment coverage
        if comment or key in preceding_content:
            keys_with_comments.append(key)
        elif key in keys_only_in_data:
            keys_without_comments.append(key)

        # Format the output line
        if value is None:
            # Section header (nested dict)
            output_lines.append(f"{indent_str}{leaf_name}:{comment}\n")
        else:
            formatted_value = format_yaml_value(value, indent_str)
            if '\n' in formatted_value:
                # Block-style value (e.g., list)
                output_lines.append(f"{indent_str}{leaf_name}:{formatted_value}{comment}\n")
            else:
                # Inline value
                output_lines.append(f"{indent_str}{leaf_name}: {formatted_value}{comment}\n")

    # Add any trailing content from template
    if trailing:
        output_lines.extend(trailing)

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    try:
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
    except IOError as e:
        error(f"Failed to write output file: {output_path}", [str(e)])
        return False

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print(f"\n{'-'*60}", file=sys.stderr)
    
    total_data_keys = len([k for k in flat_data if flat_data[k] is not None])
    total_output_keys = len([k for k in final_key_order if k in flat_data and flat_data[k] is not None])
    
    if total_data_keys == total_output_keys:
        success(f"All {total_data_keys} data values preserved in output")
    else:
        error(f"DATA LOSS: Only {total_output_keys}/{total_data_keys} data values in output!")

    stats = [
        f"Data keys (leaf):     {len([k for k in flat_data if flat_data[k] is not None])}",
        f"Data sections:        {len([k for k in flat_data if flat_data[k] is None])}",
        f"Template keys:        {len(template_order)}",
        f"Keys in both:         {len(keys_in_both)}",
        f"Keys only in data:    {len(keys_only_in_data)}",
        f"Keys only in template:{len(keys_only_in_template)}",
    ]
    
    for stat in stats:
        print(f"  {stat}", file=sys.stderr)
    
    print(f"{'-'*60}\n", file=sys.stderr)
    
    if keys_only_in_data or keys_only_in_template:
        warn("Template and data keys do not match exactly - review output carefully")
        return True  # Succeeded but with warnings
    else:
        success("Template and data keys match perfectly!")
        return True


if __name__ == "__main__":
    from pathlib import Path
    import sys

    if len(sys.argv) == 4:
        template_path = Path(sys.argv[1])
        data_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3])
    elif len(sys.argv) == 1:
        # Default paths for testing
        template_path = Path("assets/template.yaml")
        data_path = Path("configs/test.yaml")
        output_path = Path("final_config.yaml")
    else:
        error("Invalid arguments")
        print(f"\nUsage: {sys.argv[0]} <template.yaml> <data.yaml> <output.yaml>", file=sys.stderr)
        print(f"       {sys.argv[0]}  (uses default paths for testing)", file=sys.stderr)
        sys.exit(1)

    result = inject_into_template(template_path, data_path, output_path)
    sys.exit(0 if result else 1)