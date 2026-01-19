#!/bin/bash

# Configuration
MAX_RETRIES=20
LINT_CMD="cython-lint --max-line-length=120"

# Detect OS for sed compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
  SED_IN_PLACE="sed -i ''"
  SED_EXT="-E"
else
  SED_IN_PLACE="sed -i"
  SED_EXT="-r"
fi

# ==============================================================================
# FUNCTION: Process a single file
# ==============================================================================
process_file() {
    local FILE="$1"
    echo "======================================================="
    echo "Processing: $FILE"
    echo "======================================================="

    # --- PHASE 1: Simple Whitespace, Punctuation & Comments ---
    # Covers: E201, E202, W291, E231(comma), E265
    echo "1. Fixing basic punctuation and whitespace..."

    # E201: Whitespace after '[' or '('
    eval $SED_IN_PLACE "'s/\[ /\[/g'" "$FILE"
    eval $SED_IN_PLACE "'s/( /(/g'" "$FILE"

    # E202: Whitespace before ']' or ')'
    eval $SED_IN_PLACE "'s/ \]/]/g'" "$FILE"
    eval $SED_IN_PLACE "'s/ )/)/g'" "$FILE"

    # E231: Missing whitespace after ',' (Global fix is usually safe for commas)
    eval $SED_IN_PLACE $SED_EXT "'s/,([^[:space:]])/, \1/g'" "$FILE"

    # E265: Block comment should start with '# '
    eval $SED_IN_PLACE $SED_EXT "'s/^([[:space:]]*)#([^ ![:space:]])/\1# \2/g'" "$FILE"

    # W291: Trailing whitespace at end of line
    eval $SED_IN_PLACE "'s/[[:space:]]*$//'" "$FILE"


    # --- PHASE 2: Redundant Backslashes (E502) ---
    echo "2. Fixing E502 (Redundant backslash)..."
    # Logic: If bracket is open, backslash is not needed.
    E502_LINES=$($LINT_CMD "$FILE" | grep "E502" | awk -F: '{print $2}' | sort -u -nr)
    for line in $E502_LINES; do
        # Replace backslash at the very end of the line with nothing
        # We escape the backslash twice: \\\\ matches a literal \
        CMD="$SED_IN_PLACE '${line}s/\\\\$//' \"$FILE\""
        eval "$CMD"
    done


    # --- PHASE 3: Vertical Whitespace (E301, E303) ---
    echo "3. Fixing Vertical Whitespace (E301, E303)..."

    # A. E303: Too many blank lines
    for ((i=1; i<=5; i++)); do
        E303_LINES=$($LINT_CMD "$FILE" | grep "E303" | awk -F: '{print $2}' | sort -u -nr)
        if [ -z "$E303_LINES" ]; then break; fi

        for line in $E303_LINES; do
            TARGET=$((line - 1))
            CMD="$SED_IN_PLACE '${TARGET}d' \"$FILE\""
            eval "$CMD"
        done
    done

    # B. E301: Expected 1 blank line, found 0
    E301_LINES=$($LINT_CMD "$FILE" | grep "E301" | awk -F: '{print $2}' | sort -u -nr)
    for line in $E301_LINES; do
        CMD="$SED_IN_PLACE '${line}s/^/\\n/' \"$FILE\""
        eval "$CMD"
    done


    # --- PHASE 4: Line Length (E501) ---
    echo "4. Fixing E501 (Line too long > 120)..."
    E501_LINES=$($LINT_CMD "$FILE" | grep "E501" | awk -F: '{print $2}' | sort -u -nr)
    for line in $E501_LINES; do
        CMD="$SED_IN_PLACE $SED_EXT '${line}s/^(.{1,110}) (.*)$/\1 \\\\\\n    \2/' \"$FILE\""
        eval "$CMD"
    done


    # --- PHASE 5: Operator, Colon & Comment Spacing ---
    # Covers: E221, E222, E261, E231(colon)
    echo "5. Fixing Complex Spacing..."

    for ((i=1; i<=3; i++)); do
        # E231: Missing whitespace after ':' (Smart Fix)
        # We do this here instead of Phase 1 to avoid breaking slices like x[1:2]
        E231_LINES=$($LINT_CMD "$FILE" | grep "E231" | awk -F: '{print $2}' | sort -u -nr)
        if [ ! -z "$E231_LINES" ]; then
             for line in $E231_LINES; do
                # Only add space if followed by non-space.
                # This might technically hit slices if they are on the same line as a dict error,
                # but it is safer than a global replace.
                CMD="$SED_IN_PLACE $SED_EXT '${line}s/:([^[:space:]])/: \1/g' \"$FILE\""
                eval "$CMD"
             done
        fi

        # E221: Multiple spaces BEFORE operator
        E221_LINES=$($LINT_CMD "$FILE" | grep "E221" | awk -F: '{print $2}' | sort -u -nr)
        if [ ! -z "$E221_LINES" ]; then
             for line in $E221_LINES; do
                CMD="$SED_IN_PLACE $SED_EXT '${line}s/  +([=+\*/%&|^<>!-])/ \1/g' \"$FILE\""
                eval "$CMD"
             done
        fi

        # E222: Multiple spaces AFTER operator
        E222_LINES=$($LINT_CMD "$FILE" | grep "E222" | awk -F: '{print $2}' | sort -u -nr)
        if [ ! -z "$E222_LINES" ]; then
             for line in $E222_LINES; do
                CMD="$SED_IN_PLACE $SED_EXT '${line}s/([^ ])  +/\1 /g' \"$FILE\""
                eval "$CMD"
             done
        fi

        # E261: At least two spaces before inline comment
        E261_LINES=$($LINT_CMD "$FILE" | grep "E261" | awk -F: '{print $2}' | sort -u -nr)
        if [ ! -z "$E261_LINES" ]; then
             for line in $E261_LINES; do
                CMD="$SED_IN_PLACE '${line}s/ #/  #/g' \"$FILE\""
                eval "$CMD"
             done
        fi

        # Break loop if no errors found
        if [ -z "$E231_LINES" ] && [ -z "$E221_LINES" ] && [ -z "$E222_LINES" ] && [ -z "$E261_LINES" ]; then break; fi
    done


    # --- PHASE 6: Indentation (E127, E122, E128, E111) ---
    echo "6. Smart Fixing Indentation..."

    for ((i=1; i<=MAX_RETRIES; i++)); do
        # A. Fix Over-indentation (E127) -> Remove 1 space
        E127_LINES=$($LINT_CMD "$FILE" | grep "E127" | awk -F: '{print $2}' | sort -u -nr)
        if [ ! -z "$E127_LINES" ]; then
            echo "   -> Attempt $i: Reducing indentation on $(echo $E127_LINES | wc -w) lines..."
            for line in $E127_LINES; do
                CMD="$SED_IN_PLACE '${line}s/^ //' \"$FILE\""
                eval "$CMD"
            done
        fi

        # B. Fix Under-indentation (E122, E128, E111) -> Add 1 space
        UNDER_INDENT_LINES=$($LINT_CMD "$FILE" | grep -E "E122|E128|E111" | awk -F: '{print $2}' | sort -u -nr)
        if [ ! -z "$UNDER_INDENT_LINES" ]; then
            echo "   -> Attempt $i: Increasing indentation on $(echo $UNDER_INDENT_LINES | wc -w) lines..."
            for line in $UNDER_INDENT_LINES; do
                CMD="$SED_IN_PLACE '${line}s/^/ /' \"$FILE\""
                eval "$CMD"
            done
        fi

        if [ -z "$E127_LINES" ] && [ -z "$UNDER_INDENT_LINES" ]; then break; fi
    done

    echo "Done with $FILE."
}

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

TARGET="${1:-.}"

if [ -f "$TARGET" ]; then
    # User provided a single file
    process_file "$TARGET"
elif [ -d "$TARGET" ]; then
    # User provided a directory (or default .).
    # Find both .pyx and .pxd files recursively
    echo "Searching for .pyx and .pxd files in '$TARGET'..."
    find "$TARGET" -type f \( -name "*.pyx" -o -name "*.pxd" \) | while read -r found_file; do
        process_file "$found_file"
    done
else
    echo "Error: '$TARGET' is not a valid file or directory."
    exit 1
fi

echo "All tasks completed."
