# Smoke stop
This is a functional interface test, not a grasp-failure detector.
When red-high fires, inspect current front RGB and call arx.finish with reason
"RGB critic functional test completed" using the current observation and epoch.
Do not move, resume Zeva, read private state, or assert task success.
