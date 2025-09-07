# Creates a new homework assignment from the template if it doesn't exist,
# then runs it.
#
# Usage:
#   just new hw02
new hw_name:
    @if [ ! -d "{{hw_name}}" ]; then \
        echo "==> Creating new homework '{{hw_name}}' from template..."; \
        copier copy hw-template {{hw_name}}; \
        just run {{hw_name}}; \
    else \
        echo "==> Homework '{{hw_name}}' already exists. Skipping creation."; \
    fi

# Runs a specific homework assignment.
#
# Usage:
#   just run hw02
run hw_name:
    @echo "==> Running homework '{{hw_name}}'..."
    @uv run --project {{hw_name}} {{hw_name}}

# Creates a clean copy of the repository in a specified directory
# with a fresh git history.
#
# Usage: just export ../path/to/new-repo
export path:
    @if [ -d "{{path}}" ] && [ "$(ls -A '{{path}}')" ]; then \
        echo "Error: Directory '{{path}}' exists and is not empty." >&2; \
        exit 1; \
    fi
    @mkdir -p "{{path}}"
    @echo "--> Exporting a clean version of the repository to '{{path}}'..."
    @git archive HEAD | tar -x -C "{{path}}"
    @(cd "{{path}}" && \
      git init --initial-branch=main > /dev/null && \
      git add . && \
      git commit -m "Initial commit" > /dev/null)
    @echo "--> Successfully created a clean repository in '{{path}}'."
