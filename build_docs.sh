#!/bin/bash
# Script to build Doxygen documentation for Hydra Image Processor

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Hydra Image Processor - Documentation Build${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo -e "${RED}Error: Doxygen is not installed${NC}"
    echo ""
    echo "Please install Doxygen:"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen graphviz"
    echo "  macOS:         brew install doxygen graphviz"
    echo "  Windows:       Download from https://www.doxygen.nl/download.html"
    exit 1
fi

# Display Doxygen version
DOXYGEN_VERSION=$(doxygen --version)
echo -e "Using Doxygen version: ${GREEN}${DOXYGEN_VERSION}${NC}"
echo ""

# Check if Graphviz/dot is installed (optional but recommended)
if command -v dot &> /dev/null; then
    DOT_VERSION=$(dot -V 2>&1 | head -n1)
    echo -e "Using Graphviz: ${GREEN}${DOT_VERSION}${NC}"
else
    echo -e "${YELLOW}Warning: Graphviz (dot) not found. Graphs will not be generated.${NC}"
    echo "  Install with: sudo apt-get install graphviz (Linux) or brew install graphviz (macOS)"
fi
echo ""

# Check if Doxyfile exists
if [ ! -f "Doxyfile" ]; then
    echo -e "${RED}Error: Doxyfile not found in current directory${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Clean previous documentation
if [ -d "docs/html" ]; then
    echo -e "${YELLOW}Cleaning previous documentation...${NC}"
    rm -rf docs/html docs/latex docs/xml docs/man
fi

# Build documentation
echo -e "${GREEN}Building documentation...${NC}"
doxygen Doxyfile

# Check if build was successful
if [ -d "docs/html" ] && [ -f "docs/html/index.html" ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Documentation built successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "View the documentation by opening:"
    echo -e "  ${GREEN}docs/html/index.html${NC}"
    echo ""
    echo "To view in browser:"
    echo "  Linux:   xdg-open docs/html/index.html"
    echo "  macOS:   open docs/html/index.html"
    echo "  Windows: start docs/html/index.html"
    echo ""
else
    echo -e "${RED}Error: Documentation build failed${NC}"
    exit 1
fi
