"""Recorded camera-only action: restores manual exposure using active streams.

Executing this file changes D405 camera settings; it never opens ROS or CAN.
"""

def main():
    import json, time
    import pyrealsense2 as rs
    records = []
    for serial in ['260422272500', '260422271945', '260422275847']:
        pipe, config = rs.pipeline(), rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        if serial == '260422272500':
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        profile = pipe.start(config)
        try:
            for _ in range(10):
                pipe.wait_for_frames(1000)
            sensor = profile.get_device().first_depth_sensor()
            item = {'serial': serial, 'verified': {}, 'busy_responses': 0}
            for name, target in [('enable_auto_exposure', 0.0), ('exposure', 25000.0), ('gain', 16.0)]:
                opt = getattr(rs.option, name)
                deadline = time.monotonic() + 3
                while True:
                    try:
                        value = sensor.get_option(opt)
                        if value == target:
                            item['verified'][name] = value
                            break
                        sensor.set_option(opt, target)
                    except RuntimeError as error:
                        if 'busy' not in str(error).lower():
                            raise
                        item['busy_responses'] += 1
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'{serial}/{name} did not confirm {target}')
                    for _ in range(4):
                        pipe.wait_for_frames(1000)
            records.append(item)
        finally:
            pipe.stop()
    print(json.dumps({'restored_recorded_baseline': True, 'streams_active_during_configuration': True, 'devices': records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
