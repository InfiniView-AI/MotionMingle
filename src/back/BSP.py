# Body Segment Partition parameters
# All parameters are from: https://www.sciencedirect.com/science/article/abs/pii/0021929095001786?via%3Dihub
BSP = {
    "head": {
        "origin": 7,
        "other": 8,
        "l": 0.5,
        "m": (0.0668 + 0.0694) / 2
    },
    "trunk": {
        "origin": (11, 12),
        "other": (23, 24),
        "l": (0.4151 + 0.4486) / 2,
        "m": (0.4257 + 0.4346) / 2
    },
    "left_upper_arm": {
        "origin": 11,
        "other": 13,
        "l": (0.5754 + 0.5772) / 2,
        "m": (0.0255 + 0.0271) / 2
    },
    "right_upper_arm": {
        "origin": 12,
        "other": 14,
        "l": (0.5754 + 0.5772) / 2,
        "m": (0.0255 + 0.0271) / 2
    },
    "left_forearm": {
        "origin": 13,
        "other": 15,
        "l": (0.4559 + 0.4574) / 2, # length (man + woman)
        "m": (0.0138 + 0.0162) / 2
    },
    "right_forearm": {
        "origin": 14,
        "other": 16,
        "l": (0.4559 + 0.4574) / 2,
        "m": (0.0138 + 0.0162) / 2
    },
    "left_hand": {
        "origin": 15,
        "other": (17, 19),
        "l": (0.7474 + 0.7900) / 2,
        "m": (0.0056 + 0.0061) / 2
    },
    "right_hand": {
        "origin": 16,
        "other": (18, 20),
        "l": (0.7474 + 0.7900) / 2,
        "m": (0.0056 + 0.0061) / 2
    },
    "left_thigh": {
        "origin": 23,
        "other": 25,
        "l": (0.3612 + 0.4095) / 2,
        "m": (0.1478 + 0.1416) / 2
    },
    "right_thigh": {
        "origin": 24,
        "other": 26,
        "l": (0.3612 + 0.4095) / 2,
        "m": (0.1478 + 0.1416) / 2
    },
    "left_shank": {
        "origin": 25,
        "other": 27,
        "l": (0.4416 + 0.4459) / 2,
        "m": (0.0481 + 0.0433) / 2
    },
    "right_shank": {
        "origin": 26,
        "other": 28,
        "l": (0.4416 + 0.4459) / 2,
        "m": (0.0481 + 0.0433) / 2
    },
    "left_foot": {
        "origin": 29,
        "other": 31,
        "l": (0.4014 + 0.4415) / 2,
        "m": (0.0129 + 0.0137) / 2
    },
    "right_foot": {
        "origin": 30,
        "other": 32,
        "l": (0.4014 + 0.4415) / 2,
        "m": (0.0129 + 0.0137) / 2
    }
}
