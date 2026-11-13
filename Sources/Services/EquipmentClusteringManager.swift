import Combine
import Foundation
import simd

/// `@MainActor`-isolated spatial registry that clusters 3D equipment detections
/// using size-aware Euclidean distance merging and weighted running-average smoothing.
@MainActor
public final class EquipmentClusteringManager: ObservableObject {
    /// Default fallback Euclidean distance (in meters) for classes outside explicit size tiers.
    nonisolated public static let defaultMergeRadiusMeters: Float = 0.35
    /// Tighter clustering radius (0.20 m) for small desktop/handheld items so adjacent objects do not merge.
    nonisolated public static let smallItemMergeRadiusMeters: Float = 0.20
    /// Wider clustering radius (0.45 m) for large furniture and clinical equipment.
    nonisolated public static let largeItemMergeRadiusMeters: Float = 0.45

    nonisolated private static let smallItemKeywords: [String] = [
        "bottle", "cup", "mug", "mouse", "keyboard", "book",
        "pen", "mousepad", "pipette", "thermometer", "barcode scanner",
        "telephone", "tablet", "sample tray", "gloves box"
    ]

    nonisolated private static let largeItemKeywords: [String] = [
        "table", "desk", "chair", "sofa", "phoropter", "slit lamp",
        "autorefractor", "lens drawer", "optometry chair", "armchair",
        "cabinet", "filing cabinet", "shelf", "refrigerator", "whiteboard",
        "wheelchair", "examination light", "rolling stool", "radiator", "sink", "door"
    ]

    @Published public private(set) var trackedEquipment: [TrackedEquipment3D] = []

    public let defaultMergeRadius: Float

    public init(mergeRadiusMeters: Float = EquipmentClusteringManager.defaultMergeRadiusMeters) {
        self.defaultMergeRadius = max(0.0, mergeRadiusMeters)
    }

    /// Returns the size-aware clustering radius (in meters) for a detected class label.
    nonisolated public static func mergeRadius(
        for label: String,
        fallback: Float = defaultMergeRadiusMeters
    ) -> Float {
        let normalized = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if smallItemKeywords.contains(where: { normalized.contains($0) }) {
            return smallItemMergeRadiusMeters
        }
        if largeItemKeywords.contains(where: { normalized.contains($0) }) {
            return largeItemMergeRadiusMeters
        }
        return max(0.0, fallback)
    }

    /// Registers a newly projected 3D detection.
    ///
    /// If an existing record with a matching label is strictly closer than the size-aware
    /// merge radius (`0.20 m` small items, `0.45 m` large furniture/clinical machines, `0.35 m` default),
    /// its `worldPosition` is updated using an observation-weighted running average and its
    /// `observationCount` is incremented. Otherwise, a new `TrackedEquipment3D` entity is appended.
    @discardableResult
    public func registerDetection(
        label: String,
        worldPosition: simd_float3,
        timestamp: Date = Date()
    ) -> TrackedEquipment3D? {
        let trimmedLabel = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLabel.isEmpty else { return nil }
        guard worldPosition.x.isFinite,
              worldPosition.y.isFinite,
              worldPosition.z.isFinite else {
            return nil
        }

        if let matchIndex = findClosestMatchingIndex(label: trimmedLabel, worldPosition: worldPosition) {
            var existing = trackedEquipment[matchIndex]
            let previousCount = Float(max(1, existing.observationCount))
            let newCount = existing.observationCount + 1

            let smoothedPosition = (existing.worldPosition * previousCount + worldPosition) / Float(newCount)
            existing.worldPosition = smoothedPosition
            existing.observationCount = newCount
            existing.lastSeen = max(existing.lastSeen, timestamp)

            trackedEquipment[matchIndex] = existing
            return existing
        } else {
            let newItem = TrackedEquipment3D(
                id: UUID(),
                label: trimmedLabel,
                worldPosition: worldPosition,
                lastSeen: timestamp,
                observationCount: 1
            )
            trackedEquipment.append(newItem)
            return newItem
        }
    }

    /// Returns the tracked item for a given ID, if present.
    public func equipment(withID id: UUID) -> TrackedEquipment3D? {
        trackedEquipment.first { $0.id == id }
    }

    /// Clears all tracked equipment records (e.g., when resetting an AR scanning session).
    public func clearAll() {
        trackedEquipment.removeAll()
    }

    private func findClosestMatchingIndex(label: String, worldPosition: simd_float3) -> Int? {
        var bestIndex: Int?
        var bestDistance = Self.mergeRadius(for: label, fallback: defaultMergeRadius)

        for (index, item) in trackedEquipment.enumerated() {
            guard item.label.caseInsensitiveCompare(label) == .orderedSame else {
                continue
            }
            let distance = simd_distance(worldPosition, item.worldPosition)
            if distance.isFinite, distance < bestDistance {
                bestDistance = distance
                bestIndex = index
            }
        }

        return bestIndex
    }
}
