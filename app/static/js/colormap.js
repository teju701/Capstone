/**
 * colormap.js
 * ───────────
 * Cityscapes 19-class metadata, color palettes, and metric depth utilities.
 */

const CITYSCAPES_CLASSES = {
    0:  { name: "Road",          category: "Flat",         color: [128, 64, 128], isDynamic: false },
    1:  { name: "Sidewalk",      category: "Flat",         color: [244, 35, 232], isDynamic: false },
    2:  { name: "Building",      category: "Construction", color: [70, 70, 70],   isDynamic: false },
    3:  { name: "Wall",          category: "Construction", color: [102, 102, 156],isDynamic: false },
    4:  { name: "Fence",         category: "Construction", color: [190, 153, 153],isDynamic: false },
    5:  { name: "Pole",          category: "Object",       color: [153, 153, 153],isDynamic: false },
    6:  { name: "Traffic Light", category: "Object",       color: [250, 170, 30], isDynamic: false },
    7:  { name: "Traffic Sign",  category: "Object",       color: [220, 220, 0],  isDynamic: false },
    8:  { name: "Vegetation",    category: "Nature",       color: [107, 142, 35], isDynamic: false },
    9:  { name: "Terrain",       category: "Nature",       color: [152, 251, 152],isDynamic: false },
    10: { name: "Sky",           category: "Sky",          color: [70, 130, 180], isDynamic: false },
    11: { name: "Person",        category: "Human",        color: [220, 20, 60],  isDynamic: true },
    12: { name: "Rider",         category: "Human",        color: [255, 0, 0],    isDynamic: true },
    13: { name: "Car",           category: "Vehicle",      color: [0, 0, 142],    isDynamic: true },
    14: { name: "Truck",         category: "Vehicle",      color: [0, 0, 70],     isDynamic: true },
    15: { name: "Bus",           category: "Vehicle",      color: [0, 60, 100],   isDynamic: true },
    16: { name: "Train",         category: "Vehicle",      color: [0, 80, 100],   isDynamic: true },
    17: { name: "Motorcycle",    category: "Vehicle",      color: [0, 0, 230],    isDynamic: true },
    18: { name: "Bicycle",       category: "Vehicle",      color: [119, 11, 32],  isDynamic: true }
};

function getClassInfo(classId) {
    return CITYSCAPES_CLASSES[classId] || {
        name: `Class ${classId}`,
        category: "Unknown",
        color: [120, 120, 120],
        isDynamic: false
    };
}

function getSafetyStatus(distanceMeters, isDynamic) {
    if (distanceMeters < 5.0 && isDynamic) {
        return { text: "CRITICAL HAZARD", color: "#ef4444", class: "text-red" };
    } else if (distanceMeters < 15.0 && isDynamic) {
        return { text: "PROXIMITY WARNING", color: "#f59e0b", class: "text-yellow" };
    } else if (distanceMeters < 5.0) {
        return { text: "CLOSE PROXIMITY", color: "#f59e0b", class: "text-yellow" };
    } else {
        return { text: "SAFE CORRIDOR", color: "#10b981", class: "text-green" };
    }
}
