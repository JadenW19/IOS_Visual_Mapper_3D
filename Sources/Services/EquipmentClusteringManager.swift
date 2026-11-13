import Combine
import Foundation
import simd

/// `@MainActor`-isolated spatial registry that clusters 3D equipment detections
/// using Euclidean distance merging and weighted running-average smoothing.
@MainActor
public final class EquipmentClusteringManager: ObservableObject {
    /// Maximum Euclidean distance (in meters) within which matching-label detections merge.
    nonisolated public static let defaultMergeRadiusMeters: Float = 0.35

    @Published public private(set) var trackedEquipment: [TrackedEquipment3D] = []

    public let mergeRadiusMeters: Float

    public init(mergeRadiusMeters: Float = EquipmentClusteringManager.defaultMergeRadiusMeters) {
        self.mergeRadiusMeters = max(0.0, mergeRadiusMeters)
    }

    /// Registers a newly projected 3D detection.
    ///
    /// If an existing record with a matching label is strictly closer than `mergeRadiusMeters` (0.35 m),
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
        var bestDistance = mergeRadiusMeters

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
