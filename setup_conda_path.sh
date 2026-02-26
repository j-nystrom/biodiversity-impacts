#!/bin/sh
set -eu

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: CONDA_PREFIX is not set. Activate the conda env first." >&2
  exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ -d "$script_dir/core" ]; then
  project_root="$script_dir"
else
  project_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
fi

activate_dir="$CONDA_PREFIX/etc/conda/activate.d"
deactivate_dir="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$activate_dir" "$deactivate_dir"

cat > "$activate_dir/set_path.sh" <<EOF
#!/bin/sh
export _OLD_PYTHONPATH="\${PYTHONPATH-}"
if [ -n "\${PYTHONPATH-}" ]; then
  export PYTHONPATH="$project_root:\$PYTHONPATH"
else
  export PYTHONPATH="$project_root"
fi
EOF

cat > "$deactivate_dir/unset_path.sh" <<'EOF'
#!/bin/sh
export PYTHONPATH="${_OLD_PYTHONPATH-}"
unset _OLD_PYTHONPATH
EOF

echo "Wrote:"
echo "  $activate_dir/set_path.sh"
echo "  $deactivate_dir/unset_path.sh"
