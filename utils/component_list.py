from utils.component_classes import Component

def getComponents():

    # Create Components to put in spacecraft. Same as ones used in 
    # Spacecraft Component Adaptive Layout Environment (SCALE): An efficient optimization tool
    # by Fakoor
    # transferLearningComponents = [
    # componentList = [
    #     Component(type="battery", mass=8, dimensions=[.25,.2,.15], heatDisp=2, pointing=False),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2, pointing=False),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2, pointing=False),
    #     Component(type="reaction wheel", mass=2, dimensions=[.075,.240,.240], heatDisp=2, pointing=False),
    #     Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5, pointing=False),
    #     Component(type="gyro", mass=3, dimensions=[.1876,.1239,.0015], heatDisp=2.5, pointing=False),
    #     Component(type="transmitter", mass=4.5, dimensions=[.25,.15,.05], heatDisp=12, pointing=True),
    #     Component(type="transmitter", mass=3.5, dimensions=[.2,.1,.05], heatDisp=10, pointing=True),
    #     Component(type="reciever", mass=4, dimensions=[.2,.15,.03], heatDisp=11, pointing=True),
    #     Component(type="reciever", mass=3, dimensions=[.175,.125,.03], heatDisp=9, pointing=True),
    #     Component(type="PCU", mass=7, dimensions=[.3,.2,.15], heatDisp=7, pointing=False),
    #     Component(type="OBDH", mass=9, dimensions=[.24,.18,.18], heatDisp=6, pointing=False),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5, pointing=False),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5, pointing=False),
    #     Component(type="magnetometer", mass=1.5, dimensions=[.15,.12,.04], heatDisp=1.5, pointing=False),
    #     Component(type="payload", mass=5, dimensions=[.3,.25,.2], heatDisp=3, pointing=True),
    #     Component(type="solar panel", mass=1.5, dimensions=[.3,.5,.01], heatDisp=1.5, pointing=True),
    #     Component(type="solar panel", mass=1.5, dimensions=[.3,.5,.01], heatDisp=1.5, pointing=True)
    # ]
    # componentList = [
    #     # Component(type="battery", mass=8, dimensions=[.25,.2,.15], heatDisp=2),
    #     # Component(type="transmitter", mass=3.5, dimensions=[.2,.1,.05], heatDisp=10),
    #     Component(type="PCU", mass=7, dimensions=[.3,.2,.15], heatDisp=7)
    # ]
        # ],

    componentList = [
        Component(type="solar panel", mass=1.5, dimensions=[.2,.5,.01], heatDisp=0.0, pointing=True),
        Component(type="solar panel", mass=1.6, dimensions=[.21,.52,.01], heatDisp=0.0, pointing=True),
        Component(type="solar panel", mass=1.4, dimensions=[.19,.49,.01], heatDisp=0.0, pointing=True),

        Component(type="payload", mass=6.5, dimensions=[.3,.24,.22], heatDisp=3.2, pointing=True),
        Component(type="payload", mass=5.5, dimensions=[.28,.22,.2], heatDisp=3, pointing=True),

        Component(type="transmitter", mass=3.8, dimensions=[.25,.1,.08], heatDisp=12, pointing=True),
        Component(type="transmitter", mass=4.0, dimensions=[.23,.12,.09], heatDisp=11, pointing=True),

        Component(type="receiver", mass=3.3, dimensions=[.21,.12,.05], heatDisp=1.2, pointing=True),
        Component(type="receiver", mass=3.5, dimensions=[.22,.13,.06], heatDisp=1.5, pointing=True),

        Component(type="antenna", mass=4.5, dimensions=[.35,.14,.12], heatDisp=0.2, pointing=True),
        Component(type="antenna", mass=3.2, dimensions=[.24,.1,.08], heatDisp=0.2, pointing=True),
        Component(type="antenna", mass=4.0, dimensions=[.34,.15,.1], heatDisp=0.2, pointing=True),

        Component(type="star tracker", mass=1.7, dimensions=[.11,.13,.1], heatDisp=1.3, pointing=True),
        Component(type="star tracker", mass=1.8, dimensions=[.12,.14,.11], heatDisp=1.4, pointing=True),
        Component(type="star tracker", mass=1.6, dimensions=[.1,.12,.1], heatDisp=1.2, pointing=True),

        Component(type="sun sensor", mass=1.2, dimensions=[.1,.09,.08], heatDisp=0.9, pointing=True),
        Component(type="sun sensor", mass=1.1, dimensions=[.1,.08,.07], heatDisp=1, pointing=True),
        Component(type="sun sensor", mass=1.3, dimensions=[.12,.1,.09], heatDisp=1.1, pointing=True),

        Component(type="battery", mass=5.8, dimensions=[.23,.21,.13], heatDisp=2.2, pointing=False),
        Component(type="battery", mass=6.0, dimensions=[.24,.22,.14], heatDisp=2.4, pointing=False),

        Component(type="PCU", mass=7, dimensions=[.26,.2,.14], heatDisp=6.5, pointing=False),
        Component(type="PCU", mass=6.5, dimensions=[.25,.19,.13], heatDisp=6.3, pointing=False),

        Component(type="OBDH", mass=9, dimensions=[.24,.19,.16], heatDisp=5.8, pointing=False),
        Component(type="OBDH", mass=8.8, dimensions=[.23,.18,.15], heatDisp=5.7, pointing=False),

        Component(type="reaction wheel", mass=3, dimensions=[.14,.12,.1], heatDisp=8.0, pointing=False),
        Component(type="reaction wheel", mass=3.2, dimensions=[.15,.13,.11], heatDisp=8.5, pointing=False),

        Component(type="propellant tank", mass=13, dimensions=[.3,.25,.2], heatDisp=0.0, pointing=False),
        Component(type="propellant tank", mass=12.5, dimensions=[.29,.24,.19], heatDisp=0.0, pointing=False),

        Component(type="attitude thruster", mass=2.5, dimensions=[.15,.14,.12], heatDisp=4.8, pointing=True),
        Component(type="attitude thruster", mass=2.7, dimensions=[.16,.15,.13], heatDisp=5, pointing=True),

        Component(type="IMU", mass=2.5, dimensions=[.14,.12,.09], heatDisp=5.0, pointing=False),
        Component(type="IMU", mass=2.4, dimensions=[.13,.11,.08], heatDisp=4.8, pointing=False),

        Component(type="atomic clock", mass=1.8, dimensions=[.12,.11,.07], heatDisp=10.0, pointing=False),

        Component(type="heater", mass=1.2, dimensions=[.09,.07,.05], heatDisp=1.9, pointing=False),
        Component(type="heater", mass=1.3, dimensions=[.1,.08,.06], heatDisp=2, pointing=False),

        Component(type="gyro", mass=3.1, dimensions=[.19,.13,.02], heatDisp=2.8, pointing=False),
        Component(type="gyro", mass=3.0, dimensions=[.18,.12,.02], heatDisp=2.7, pointing=False),

        Component(type="magnetometer", mass=1.4, dimensions=[.1,.12,.1], heatDisp=1.1, pointing=False),
        Component(type="magnetometer", mass=1.5, dimensions=[.11,.13,.09], heatDisp=1.2, pointing=False),

        Component(type="accelerometer", mass=1.8, dimensions=[.13,.11,.09], heatDisp=2.2, pointing=False),
        Component(type="accelerometer", mass=1.7, dimensions=[.12,.1,.08], heatDisp=2.1, pointing=False)
    ]

    transferLearningComponents = [
        [
            Component(type="solar panel", mass=1.8, dimensions=[0.22, 0.53, 0.012], heatDisp=1.7, pointing=True),
            Component(type="solar panel", mass=1.5, dimensions=[0.20, 0.48, 0.01], heatDisp=1.5, pointing=True),
            Component(type="solar panel", mass=1.7, dimensions=[0.23, 0.52, 0.011], heatDisp=1.6, pointing=True),
            Component(type="payload", mass=6.8, dimensions=[0.31, 0.25, 0.22], heatDisp=3.3, pointing=True),
            Component(type="payload", mass=5.9, dimensions=[0.29, 0.24, 0.21], heatDisp=3.2, pointing=True),
            Component(type="transmitter", mass=4.2, dimensions=[0.26, 0.12, 0.09], heatDisp=12.5, pointing=True),
            Component(type="transmitter", mass=3.9, dimensions=[0.25, 0.11, 0.08], heatDisp=11.3, pointing=True),
            Component(type="receiver", mass=3.4, dimensions=[0.22, 0.13, 0.06], heatDisp=9.7, pointing=True),
            Component(type="receiver", mass=3.7, dimensions=[0.24, 0.14, 0.07], heatDisp=10, pointing=True),
            Component(type="antenna", mass=4.2, dimensions=[0.36, 0.15, 0.11], heatDisp=10, pointing=True),
            Component(type="antenna", mass=3.8, dimensions=[0.30, 0.12, 0.09], heatDisp=9, pointing=True),
            Component(type="antenna", mass=4.5, dimensions=[0.37, 0.16, 0.12], heatDisp=10.2, pointing=True),
            Component(type="star tracker", mass=1.9, dimensions=[0.13, 0.15, 0.11], heatDisp=1.5, pointing=True),
            Component(type="star tracker", mass=1.6, dimensions=[0.11, 0.13, 0.10], heatDisp=1.3, pointing=True),
            Component(type="star tracker", mass=1.8, dimensions=[0.12, 0.14, 0.11], heatDisp=1.4, pointing=True),
            Component(type="sun sensor", mass=1.3, dimensions=[0.11, 0.10, 0.08], heatDisp=1.2, pointing=True),
            Component(type="sun sensor", mass=1.4, dimensions=[0.12, 0.11, 0.09], heatDisp=1.3, pointing=True),
            Component(type="sun sensor", mass=1.2, dimensions=[0.10, 0.09, 0.07], heatDisp=1.1, pointing=True),
            Component(type="battery", mass=6.3, dimensions=[0.25, 0.23, 0.15], heatDisp=2.6, pointing=False),
            Component(type="battery", mass=6.1, dimensions=[0.24, 0.22, 0.14], heatDisp=2.5, pointing=False),
            Component(type="PCU", mass=6.9, dimensions=[0.27, 0.21, 0.15], heatDisp=6.7, pointing=False),
            Component(type="PCU", mass=6.6, dimensions=[0.26, 0.20, 0.14], heatDisp=6.4, pointing=False),
            Component(type="OBDH", mass=9.5, dimensions=[0.26, 0.20, 0.17], heatDisp=6.0, pointing=False),
            Component(type="OBDH", mass=8.9, dimensions=[0.25, 0.19, 0.16], heatDisp=5.9, pointing=False),
            Component(type="reaction wheel", mass=3.4, dimensions=[0.15, 0.13, 0.11], heatDisp=3.5, pointing=False),
            Component(type="reaction wheel", mass=3.1, dimensions=[0.14, 0.12, 0.10], heatDisp=3.3, pointing=False),
            Component(type="propellant tank", mass=13.2, dimensions=[0.31, 0.26, 0.21], heatDisp=4.3, pointing=False),
            Component(type="propellant tank", mass=12.8, dimensions=[0.30, 0.25, 0.20], heatDisp=4.2, pointing=False),
            Component(type="attitude thruster", mass=2.6, dimensions=[0.16, 0.15, 0.13], heatDisp=5.2, pointing=True),
            Component(type="attitude thruster", mass=2.8, dimensions=[0.17, 0.16, 0.14], heatDisp=5.3, pointing=True),
            Component(type="IMU", mass=2.7, dimensions=[0.15, 0.13, 0.10], heatDisp=2.6, pointing=False),
            Component(type="IMU", mass=2.5, dimensions=[0.14, 0.12, 0.09], heatDisp=2.5, pointing=False),
            Component(type="atomic clock", mass=1.9, dimensions=[0.13, 0.12, 0.08], heatDisp=1.8, pointing=False),
            Component(type="heater", mass=1.4, dimensions=[0.11, 0.09, 0.06], heatDisp=2.1, pointing=False),
            Component(type="heater", mass=1.3, dimensions=[0.10, 0.08, 0.05], heatDisp=2.0, pointing=False),
            Component(type="gyro", mass=3.2, dimensions=[0.20, 0.14, 0.03], heatDisp=3.0, pointing=False),
            Component(type="gyro", mass=3.3, dimensions=[0.19, 0.13, 0.03], heatDisp=2.9, pointing=False),
            Component(type="magnetometer", mass=1.6, dimensions=[0.11, 0.13, 0.11], heatDisp=1.3, pointing=False),
            Component(type="magnetometer", mass=1.5, dimensions=[0.10, 0.12, 0.10], heatDisp=1.2, pointing=False),
            Component(type="accelerometer", mass=1.9, dimensions=[0.14, 0.12, 0.10], heatDisp=2.4, pointing=False),
            Component(type="accelerometer", mass=1.8, dimensions=[0.13, 0.11, 0.09], heatDisp=2.3, pointing=False),
        ]
    ]

    return componentList, transferLearningComponents


import random

def create_varied_components(num_sets=10, variation_pct=0.3):
    """
    Creates a new component list where all components of the same type
    share the same mean values, with each instance varied around that mean.

    Args:
        num_sets (int): The number of varied component sets to create.
        variation_pct (float): The max percentage variation to apply (default 10%)

    Returns:
        list: A new list of Component objects with varied properties.
    """

    def vary(value):
        """Apply a small random variation to a single numeric value."""
        factor = 1 + random.uniform(-variation_pct, variation_pct)
        return round(value * factor, 4)

    def vary_dims(dims):
        """Apply variation to each dimension in a list."""
        return [vary(d) for d in dims]

    # Mean values per component type, derived from componentList [1]
    means = {
        "solar panel":       {"mass": 1.5,  "dims": [0.20,  0.50,  0.010], "heatDisp": 1.5},
        "payload":           {"mass": 6.0,  "dims": [0.29,  0.23,  0.21],  "heatDisp": 3.1},
        "transmitter":       {"mass": 3.9,  "dims": [0.24,  0.11,  0.085], "heatDisp": 11.5},
        "receiver":          {"mass": 3.4,  "dims": [0.215, 0.125, 0.055], "heatDisp": 9.25},
        "antenna":           {"mass": 3.9,  "dims": [0.31,  0.13,  0.10],  "heatDisp": 8.9},
        "star tracker":      {"mass": 1.7,  "dims": [0.11,  0.13,  0.10],  "heatDisp": 1.3},
        "sun sensor":        {"mass": 1.2,  "dims": [0.107, 0.09,  0.08],  "heatDisp": 1.0},
        "battery":           {"mass": 5.9,  "dims": [0.235, 0.215, 0.135], "heatDisp": 2.3},
        "PCU":               {"mass": 6.75, "dims": [0.255, 0.195, 0.135], "heatDisp": 6.4},
        "OBDH":              {"mass": 8.9,  "dims": [0.235, 0.185, 0.155], "heatDisp": 5.75},
        "reaction wheel":    {"mass": 3.1,  "dims": [0.145, 0.125, 0.105], "heatDisp": 3.25},
        "propellant tank":   {"mass": 12.75,"dims": [0.295, 0.245, 0.195], "heatDisp": 4.15},
        "attitude thruster": {"mass": 2.6,  "dims": [0.155, 0.145, 0.125], "heatDisp": 4.9},
        "IMU":               {"mass": 2.45, "dims": [0.135, 0.115, 0.085], "heatDisp": 2.45},
        "atomic clock":      {"mass": 1.8,  "dims": [0.12,  0.11,  0.07],  "heatDisp": 1.6},
        "heater":            {"mass": 1.25, "dims": [0.095, 0.075, 0.055], "heatDisp": 1.95},
        "gyro":              {"mass": 3.05, "dims": [0.185, 0.125, 0.02],  "heatDisp": 2.75},
        "magnetometer":      {"mass": 1.45, "dims": [0.105, 0.125, 0.095], "heatDisp": 1.15},
        "accelerometer":     {"mass": 1.75, "dims": [0.125, 0.105, 0.085], "heatDisp": 2.15},
    }

    def make(type_, pointing):
        """Helper to create a varied component from its type mean."""
        m = means[type_]
        return Component(
            type=type_,
            mass=vary(m["mass"]),
            dimensions=vary_dims(m["dims"]),
            heatDisp=vary(m["heatDisp"]),
            pointing=pointing
        )

    all_sets = []
    for i in range(num_sets):
        varied_components = [
            # Solar panels
            make("solar panel",       True),
            make("solar panel",       True),
            make("solar panel",       True),
            # Payloads
            make("payload",           True),
            make("payload",           True),
            # Transmitters
            make("transmitter",       True),
            make("transmitter",       True),
            # Receivers
            make("receiver",          True),
            make("receiver",          True),
            # Antennas
            make("antenna",           True),
            make("antenna",           True),
            make("antenna",           True),
            # Star trackers
            make("star tracker",      True),
            make("star tracker",      True),
            make("star tracker",      True),
            # Sun sensors
            make("sun sensor",        True),
            make("sun sensor",        True),
            make("sun sensor",        True),
            # Batteries
            make("battery",           False),
            make("battery",           False),
            # PCUs
            make("PCU",               False),
            make("PCU",               False),
            # OBDHs
            make("OBDH",              False),
            make("OBDH",              False),
            # Reaction wheels
            make("reaction wheel",    False),
            make("reaction wheel",    False),
            # Propellant tanks
            make("propellant tank",   False),
            make("propellant tank",   False),
            # Attitude thrusters
            make("attitude thruster", True),
            make("attitude thruster", True),
            # IMUs
            make("IMU",               False),
            make("IMU",               False),
            # Atomic clock
            make("atomic clock",      False),
            # Heaters
            make("heater",            False),
            make("heater",            False),
            # Gyros
            make("gyro",              False),
            make("gyro",              False),
            # Magnetometers
            make("magnetometer",      False),
            make("magnetometer",      False),
            # Accelerometers
            make("accelerometer",     False),
            make("accelerometer",     False),
        ]
        all_sets.append(varied_components)

    return all_sets
