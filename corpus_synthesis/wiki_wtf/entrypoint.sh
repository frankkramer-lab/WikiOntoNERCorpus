#!/bin/bash

SERVER_PIDS=()
_term() {
    echo "Caught SIGTERM signal!"
    for pid in ${SERVER_PIDS[@]}; do
        kill -TERM "$pid" 2>/dev/null
    done
}
trap _term SIGTERM

N_WORKERS="${N_WORKERS:-2}"
SERVER_PORT="${SERVER_PORT:-3000}"

launch_server() {
    SERVER_PORT=$1 node "./dist/app.js" &
    SERVER_PIDS+=("$!")
}

for ((i=0;i<N_WORKERS;i++)); do
    THIS_SERVER_PORT=$((SERVER_PORT+i))
    launch_server "$THIS_SERVER_PORT"
    echo "Starting server at port $THIS_SERVER_PORT"
done

# Wait till all processes terminated
for pid in ${SERVER_PIDS[@]}; do
    wait "$pid"
done