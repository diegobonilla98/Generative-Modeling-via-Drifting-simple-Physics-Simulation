#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v node &> /dev/null; then
    echo "Node.js is not installed or not in PATH"
    echo ""
    echo "Please install Node.js:"
    echo "  1. Download from: https://nodejs.org/"
    echo "  2. Or use your package manager (apt, yum, brew, etc.)"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed or not in PATH"
    exit 1
fi

PACKAGE_JSON="$SCRIPT_DIR/package.json"
if [ ! -f "$PACKAGE_JSON" ]; then
    echo "Error: package.json not found"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo "Installing dependencies..."
    cd "$SCRIPT_DIR"
    npm install
fi

echo "Starting development server..."
cd "$SCRIPT_DIR"
npm run dev
