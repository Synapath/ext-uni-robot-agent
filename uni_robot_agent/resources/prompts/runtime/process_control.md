Manage only computation processes you started. To stop a slow local computation,
prefer the native terminal tool's cancellation mechanism for its exact returned
session or job. If cancellation requires a signal, use the specific PID recorded
when you launched it and verify that it still belongs to that computation.
Do not use broad pkill/killall matches such as python or python3, or signal
shared process groups, harness services or other tasks. Your working directory
does not restrict which processes a kill command can affect. If ownership is
unclear, do not kill it. Stopping a script does not cancel robot actions already
submitted; inspect actual state before continuing and never replay them blindly.
