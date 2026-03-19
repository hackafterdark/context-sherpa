# Building Context Sherpa

Context Sherpa can be built in two modes: the **GUI Application** (powered by Wails) and the **Headless MCP Server** (standard Go binary).

## Prerequisites

- **Go**: Version 1.21 or later.
- **Node.js**: Version 20 or later (required for GUI/Frontend).
- **Wails CLI**: Required for building the GUI.
  ```bash
  go install github.com/wailsapp/wails/v2/cmd/wails@latest
  ```

---

The GUI version provides the full visual experience, including the Code Atlas, Agent Rule Editor, and easy dependency management. For more advanced build scenarios, refer to the [Wails Manual Builds Guide](https://wails.io/docs/guides/manual-builds).

### Frontend Build (Required)

Before building the GUI with Wails, you must first build the React Vite application in the `frontend` directory:

```bash
cd frontend
npm install
npm run build
cd ..
```

### Development Mode
Runs the application with a live-reloading frontend.
```bash
wails dev
```

### Production Build
Generates a platform-native executable in `build/bin/`.
```bash
wails build
```

---

## Building the Headless MCP Server

The headless version is a standalone binary that provides only the MCP server interface. Perfect for lightweight installations or CI/CD integration.

Note: If you are using the GUI app, you do not need to build the headless version. The GUI app includes the MCP server. Have your AI agent use the GUI app as the MCP server.

### Build Command
Run this from the project root:
```bash
go build -o context-sherpa ./cmd/context-sherpa
```

### Running the Server
```bash
./context-sherpa --projectRoot="/path/to/your/project"
```

---

## Continuous Integration

We use GitHub Actions to automate releases:
- `.github/workflows/release.yml`: Builds and releases the **Headless** binaries.
- `.github/workflows/release-gui.yml`: Builds and releases the **GUI** applications for Windows, macOS, and Linux.
