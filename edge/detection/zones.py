"""
Zone detection for Eyes of Horus.

Checks if a detected person's position falls within named zones
defined by the camera operator (e.g., "entrance", "restricted", "loading dock").

Zones are polygons stored as lists of [x, y] points in the camera profile.
"""


def point_in_polygon(point, polygon):
    """Ray casting algorithm to check if a point is inside a polygon.

    Args:
        point: (x, y) tuple
        polygon: list of [x, y] points defining the polygon

    Returns:
        bool: True if point is inside the polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def check_zones(bbox_center, zones):
    """Check which named zones a point falls within.

    Args:
        bbox_center: (x, y) tuple — center of the person's bounding box
        zones: list of zone dicts, each with:
            - "name": str (e.g., "entrance", "restricted")
            - "points": list of [x, y] coordinates defining the polygon
            - "type": str ("alert" | "monitor" | "ignore") — optional

    Returns:
        list of zone names the point is inside
    """
    if not zones:
        return []

    matched = []
    for zone in zones:
        name = zone.get("name", "unnamed")
        points = zone.get("points", [])
        if len(points) < 3:
            continue
        if point_in_polygon(bbox_center, points):
            matched.append(name)

    return matched


def get_zone_details(bbox_center, zones):
    """Like check_zones but returns full zone info (name + type).

    Returns:
        list of dicts: [{"name": "restricted", "type": "alert"}, ...]
    """
    if not zones:
        return []

    matched = []
    for zone in zones:
        name = zone.get("name", "unnamed")
        points = zone.get("points", [])
        zone_type = zone.get("type", "monitor")
        if len(points) < 3:
            continue
        if point_in_polygon(bbox_center, points):
            matched.append({"name": name, "type": zone_type})

    return matched
