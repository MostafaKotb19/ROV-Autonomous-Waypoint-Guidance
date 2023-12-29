scenario = {
    "name": "Hovering",           
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 200,
    "frames_per_sec": False,
    "agents":[
        {
            "agent_name": "auv0",  
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "PoseSensor",
                    "socket": "IMUSocket",
                    "Hz": 200
                },
                {
                    "sensor_type": "VelocitySensor",
                    "socket": "IMUSocket",
                    "Hz": 200
                },
                {
                    "sensor_type": "RotationSensor",
                    "socket": "IMUSocket",
                    "Hz": 200
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "HorizontalRangeSensor",
                    "socket": "COM",
                    "Hz": 200,
                    "configuration": {
                        "LaserCount":8,
                        "LaserDebug":False
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "UpRangeSensor",
                    "socket": "COM",
                    "Hz": 200,
                    "configuration": {
                        "LaserCount":1,
                        "LaserAngle":90,
                        "LaserDebug":False
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "DownRangeSensor",
                    "socket": "COM",
                    "Hz": 200,
                    "configuration": {
                        "LaserCount":1,
                        "LaserAngle":-90,
                        "LaserDebug":False
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "UpInclinedRangeSensor",
                    "socket": "COM",
                    "Hz": 200,
                    "configuration": {
                        "LaserCount":2,
                        "LaserAngle":45,
                        "LaserDebug":False
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "DownInclinedRangeSensor",
                    "socket": "COM",
                    "Hz": 200,
                    "configuration": {
                        "LaserCount":2,
                        "LaserAngle":-45,
                        "LaserDebug":False
                    }
                }
            ],
            "control_scheme": 0,
            "location": [200, 200, -250],
            "rotation": [0.0, 0.0, 130.0]
        }
    ],

    "window_width":  1280,
    "window_height": 720
}