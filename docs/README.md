# Hydra Image Processor Documentation

This directory contains the generated API documentation for the Hydra Image Processor library.

## Building the Documentation

To build the documentation, you need to have [Doxygen](https://www.doxygen.nl/) installed on your system.

### Installing Doxygen

**On Ubuntu/Debian:**
```bash
sudo apt-get install doxygen graphviz
```

**On macOS:**
```bash
brew install doxygen graphviz
```

**On Windows:**
Download and install from [doxygen.nl/download.html](https://www.doxygen.nl/download.html)

### Generating Documentation

From the root directory of the project, run:

```bash
doxygen Doxyfile
```

This will generate HTML documentation in the `docs/html/` directory.

### Viewing the Documentation

Open `docs/html/index.html` in your web browser:

```bash
# On Linux
xdg-open docs/html/index.html

# On macOS
open docs/html/index.html

# On Windows
start docs/html/index.html
```

## Documentation Structure

The generated documentation includes:

- **Class List**: All classes, structs, and unions
- **File List**: All documented source files
- **Namespace List**: Organized by namespace
- **Functions**: All documented functions
- **Variables**: All documented variables and macros
- **Type Definitions**: Custom types and typedefs

## Hosting on GitHub Pages

To host this documentation on GitHub Pages:

1. Build the documentation locally:
   ```bash
   doxygen Doxyfile
   ```

2. The generated HTML will be in `docs/html/`

3. You can either:
   - **Option A**: Configure GitHub Pages to serve from the `docs/` folder
   - **Option B**: Copy the contents of `docs/html/` to a `gh-pages` branch

### Option A: Serve from docs/ folder

1. In your repository settings on GitHub, go to "Pages"
2. Under "Source", select the branch (e.g., `main`) and `/docs` folder
3. Save the settings
4. Your documentation will be available at `https://[username].github.io/[repository]/html/`

### Option B: Using gh-pages branch

```bash
# After building the docs
cd docs/html
git init
git add .
git commit -m "Update documentation"
git branch -M gh-pages
git remote add origin [your-repo-url]
git push -f origin gh-pages
```

Then configure GitHub Pages to use the `gh-pages` branch.

## Customizing the Documentation

The documentation is configured through the `Doxyfile` in the root directory. Key settings include:

- **PROJECT_NAME**: The name shown in the documentation
- **PROJECT_BRIEF**: Short description
- **INPUT**: Directories to document (currently `src/c`)
- **OUTPUT_DIRECTORY**: Where to generate docs (currently `docs`)
- **GENERATE_HTML**: Whether to generate HTML output
- **GENERATE_LATEX**: Whether to generate LaTeX/PDF output

For more information on Doxygen configuration, see the [Doxygen Manual](https://www.doxygen.nl/manual/).

## Direct Linking

Once generated, you can link directly to specific documentation pages:

- Main page: `docs/html/index.html`
- Files: `docs/html/files.html`
- Classes: `docs/html/annotated.html`
- Functions: `docs/html/globals.html`

The search functionality in the generated HTML is client-side and works without a server.
