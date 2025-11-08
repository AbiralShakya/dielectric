#!/bin/bash

# BuildZoom AI - Server Management Script
# Usage: ./restart_servers.sh [start|stop|restart]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_SERVER_DIR="$SCRIPT_DIR/buildzoom-sd-server"
BACKEND_DIR="$SCRIPT_DIR/buildzoom-ai-backend"
FRONTEND_DIR="$SCRIPT_DIR/buildzoom-ai"

stop_servers() {
    echo "🛑 Stopping all BuildZoom AI servers..."
    
    # Kill SD server (Python)
    pkill -f "python.*sd_server.py" 2>/dev/null
    pkill -f "uvicorn.*sd_server" 2>/dev/null
    
    # Kill backend (Node.js on port 3002)
    lsof -ti:3002 | xargs kill -9 2>/dev/null
    pkill -f "node.*dist/index.js" 2>/dev/null
    
    # Kill frontend (Vite on port 5173)
    lsof -ti:5173 | xargs kill -9 2>/dev/null
    pkill -f "vite" 2>/dev/null
    
    echo "✅ All servers stopped"
    sleep 1
}

start_servers() {
    echo "🚀 Starting BuildZoom AI servers..."
    
    # Check if servers are already running
    if lsof -ti:8000 >/dev/null 2>&1 || lsof -ti:3002 >/dev/null 2>&1 || lsof -ti:5173 >/dev/null 2>&1; then
        echo "⚠️  Some servers appear to be running. Use './restart_servers.sh stop' first, or './restart_servers.sh restart'"
        exit 1
    fi
    
    # Start SD Server in new terminal window
    echo "📦 Starting Stable Diffusion server..."
    osascript -e "tell app \"Terminal\" to do script \"cd '$SD_SERVER_DIR' && source venv/bin/activate && python sd_server.py\"" >/dev/null 2>&1
    
    # Wait a bit for SD server to start
    sleep 3
    
    # Start Backend in new terminal window
    echo "🔧 Starting Backend API..."
    osascript -e "tell app \"Terminal\" to do script \"cd '$BACKEND_DIR' && PORT=3002 npm start\"" >/dev/null 2>&1
    
    # Wait a bit for backend to start
    sleep 2
    
    # Start Frontend in new terminal window
    echo "🎨 Starting Frontend..."
    osascript -e "tell app \"Terminal\" to do script \"cd '$FRONTEND_DIR' && npm run dev\"" >/dev/null 2>&1
    
    echo ""
    echo "✅ All servers starting in separate terminal windows!"
    echo ""
    echo "📱 Frontend:    http://localhost:5173"
    echo "🔗 Backend:     http://localhost:3002"
    echo "🎨 SD Server:   http://localhost:8000"
    echo ""
    echo "⏳ Waiting for servers to initialize (10 seconds)..."
    sleep 10
    
    # Check server health
    echo ""
    echo "🔍 Checking server health..."
    
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ SD Server: Healthy"
    else
        echo "⚠️  SD Server: Not responding (may still be loading model)"
    fi
    
    if curl -s http://localhost:3002/health >/dev/null 2>&1; then
        echo "✅ Backend: Healthy"
    else
        echo "⚠️  Backend: Not responding"
    fi
    
    echo ""
    echo "🎉 Ready to demo! Open http://localhost:5173"
}

restart_servers() {
    stop_servers
    start_servers
}

# Main logic
case "${1:-restart}" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    restart)
        restart_servers
        ;;
    *)
        echo "Usage: $0 [start|stop|restart]"
        echo "  start   - Start all servers"
        echo "  stop    - Stop all servers"
        echo "  restart - Stop and start all servers (default)"
        exit 1
        ;;
esac

