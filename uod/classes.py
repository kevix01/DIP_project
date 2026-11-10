"""Centralized unified class mapping.

Single source of truth for mapping original COCO classes to unified classes:
- 0 (person) -> 0 (pedestrian)
- 1 (bicycle), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck) -> 2 (vehicle)
- Everything else -> itself (unchanged)
"""

# COCO class IDs for vehicle family
VEHICLE_COCO_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

# Unified class IDs
UNIFIED_PEDESTRIAN = 0
UNIFIED_VEHICLE = 2

# Mapping from COCO class to unified class
COCO_TO_UNIFIED = {
    0: UNIFIED_PEDESTRIAN,   # person -> pedestrian
    1: UNIFIED_VEHICLE,      # bicycle -> vehicle
    2: UNIFIED_VEHICLE,      # car -> vehicle
    3: UNIFIED_VEHICLE,      # motorcycle -> vehicle
    5: UNIFIED_VEHICLE,      # bus -> vehicle
    7: UNIFIED_VEHICLE,      # truck -> vehicle
}

def to_unified(coco_cls: int) -> int:
    """Map a COCO class ID to unified class (0=pedestrian, 2=vehicle)."""
    return COCO_TO_UNIFIED.get(int(coco_cls), int(coco_cls))


def is_vehicle(coco_cls: int) -> bool:
    """Check if a COCO class is a vehicle."""
    return int(coco_cls) in VEHICLE_COCO_CLASSES