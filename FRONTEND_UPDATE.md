# Professional Frontend Update

## New Professional Frontend

A completely redesigned frontend inspired by Slack and professional PCB software (Altium/KiCad).

### Features

- **Dark Professional Theme**: Clean, modern dark interface
- **Slack-Inspired Sidebar**: Organized, minimal sidebar
- **PCB Software Aesthetics**: Professional color scheme and typography
- **Clean Visualizations**: High-quality PCB plots with proper styling
- **No Emojis**: Professional, engineer-focused interface

### How to Run

**Option 1: Professional Frontend (Recommended)**
```bash
./venv/bin/streamlit run frontend/app_professional.py --server.port 8501
```

**Option 2: Original Frontend**
```bash
./venv/bin/streamlit run frontend/app.py --server.port 8501
```

### Design Philosophy

- **Minimal**: Clean, uncluttered interface
- **Professional**: Dark theme like PCB software
- **Functional**: Everything engineers need, nothing they don't
- **Modern**: Contemporary design patterns

### Color Scheme

- Background: `#1e1e1e` (dark gray)
- Sidebar: `#252526` (slightly lighter)
- Accent: `#4ec9b0` (cyan - for highlights)
- Primary: `#007acc` (blue - for buttons)
- Text: `#e0e0e0` (light gray)
- Borders: `#3e3e42` (subtle gray)

### iPhone Speaker & Siri Example

New example design available:
- **Multi-state hierarchy**: Audio Processing, Speaker Driver, Microphone Array, Siri Processing, Power Management
- **Complex routing**: Multiple power domains, audio signals, ML processing
- **Real-world complexity**: Models actual iPhone audio system architecture

Load it from the Examples dropdown in the sidebar.

