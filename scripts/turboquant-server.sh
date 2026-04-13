#!/usr/bin/env bash
# turboquant-server.sh v1.0 — TurboQuant MLX server management
# Usage: turboquant-server.sh {start|stop|status|restart}

set -euo pipefail

# Configurable via env vars
TURBOQUANT_PORT="${TURBOQUANT_PORT:-8077}"
TURBOQUANT_MODEL="${TURBOQUANT_MODEL:-mlx-community/Qwen2.5-72B-Instruct-4bit}"
TURBOQUANT_HOME="${HOME}/.turboquant"
TURBOQUANT_REPO="/Users/pdewost/Documents/Personnel/Developpement/turboquant_mlx"

PID_FILE="${TURBOQUANT_HOME}/server.pid"
LOG_FILE="${TURBOQUANT_HOME}/logs/server.log"
ERR_FILE="${TURBOQUANT_HOME}/logs/server_error.log"

mkdir -p "${TURBOQUANT_HOME}/logs"

_is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

_check_port() {
    lsof -i :"$TURBOQUANT_PORT" -sTCP:LISTEN >/dev/null 2>&1
}

cmd_start() {
    if _is_running; then
        echo "TurboQuant server already running (PID $(cat "$PID_FILE"), port $TURBOQUANT_PORT)"
        return 0
    fi

    if _check_port; then
        echo "ERROR: Port $TURBOQUANT_PORT already in use by another process"
        lsof -i :"$TURBOQUANT_PORT" -sTCP:LISTEN 2>/dev/null
        return 1
    fi

    # Pre-flight: check model exists in HF cache
    local model_dir="${HOME}/.cache/huggingface/hub/models--$(echo "$TURBOQUANT_MODEL" | sed 's|/|--|g')"
    if [ ! -d "$model_dir" ]; then
        echo "WARNING: Model cache not found at $model_dir"
        echo "First start will download the model (~40 GB). This may take a while."
    fi

    echo "Starting TurboQuant server..."
    echo "  Model: $TURBOQUANT_MODEL"
    echo "  Port:  $TURBOQUANT_PORT"
    echo "  Log:   $LOG_FILE"

    cd "$TURBOQUANT_REPO"
    nohup python3 scripts/run_server.py \
        --model "$TURBOQUANT_MODEL" \
        --port "$TURBOQUANT_PORT" \
        >> "$LOG_FILE" 2>> "$ERR_FILE" &

    local pid=$!
    echo "$pid" > "$PID_FILE"
    echo "Server started (PID $pid)"

    # Wait for port to become available (max 120s for model load)
    local waited=0
    while ! _check_port && [ $waited -lt 120 ]; do
        sleep 2
        waited=$((waited + 2))
        # Check process still alive
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: Server process died. Check $ERR_FILE"
            rm -f "$PID_FILE"
            return 1
        fi
    done

    if _check_port; then
        echo "Server ready on http://localhost:$TURBOQUANT_PORT"
    else
        echo "WARNING: Server started but port $TURBOQUANT_PORT not yet responding (model may still be loading)"
    fi
}

cmd_stop() {
    if ! _is_running; then
        echo "TurboQuant server is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    echo "Stopping TurboQuant server (PID $pid)..."
    kill "$pid" 2>/dev/null || true

    # Wait for clean shutdown (max 10s)
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ $waited -lt 10 ]; do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "Force-killing server..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "Server stopped"
}

cmd_status() {
    if _is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo "TurboQuant server RUNNING"
        echo "  PID:   $pid"
        echo "  Port:  $TURBOQUANT_PORT"
        echo "  Model: $TURBOQUANT_MODEL"

        # Memory usage
        local rss
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "$rss" ]; then
            echo "  Memory: $(( rss / 1024 )) MB"
        fi

        # API check
        if _check_port; then
            echo "  API:   responding"
        else
            echo "  API:   port not responding (model loading?)"
        fi
    else
        echo "TurboQuant server NOT RUNNING"
        rm -f "$PID_FILE"
    fi
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "Environment variables:"
        echo "  TURBOQUANT_PORT   Server port (default: 8077)"
        echo "  TURBOQUANT_MODEL  HuggingFace model ID (default: mlx-community/Qwen2.5-72B-Instruct-4bit)"
        exit 1
        ;;
esac
