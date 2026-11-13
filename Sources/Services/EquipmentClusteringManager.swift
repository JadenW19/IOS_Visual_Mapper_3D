import Combine
import Foundation
import simd

/// `@MainActor`-isolated spatial registry that clusters 3D equipment detections
/// using size-aware Euclidean distance merging, confidence-weighted multi-frame label voting,
/// and 3D bounding-box size + spatial-context disambiguation.
@MainActor
public final class EquipmentClusteringManager: ObservableObject {
    /// Default fallback Euclidean distance (in meters) for classes outside explicit size tiers.
    nonisolated public static let defaultMergeRadiusMeters: Float = 0.35
    /// Tighter clustering radius (0.20 m) for small desktop/handheld items so adjacent objects do not merge.
    nonisolated public static let smallItemMergeRadiusMeters: Float = 0.20
    /// Wider clustering radius (0.45 m) for large furniture and clinical equipment.
    nonisolated public static let largeItemMergeRadiusMeters: Float = 0.45

    /// Maximum physical width (meters) below which a detected `television` is reclassified as `monitor`.
    nonisolated public static let televisionMaxMonitorWidthMeters: Float = 0.75
    /// Proximity radius (meters) within which desktop context (`desk`, `keyboard`, `mouse`, `laptop`)
    /// reclassifies a nearby `television` as `monitor`.
    nonisolated public static let monitorContextRadiusMeters: Float = 0.80
    /// Proximity radius (meters) within which `monitor`, `laptop`, or `keyboard` above/on a `table`
    /// reclassifies that `table` as `desk`.
    nonisolated public static let deskContextRadiusMeters: Float = 0.60

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

    nonisolated private static let monitorContextLabels: Set<String> = [
        "desk", "keyboard", "mouse", "laptop"
    ]

    nonisolated private static let deskContextLabels: Set<String> = [
        "monitor", "laptop", "keyboard"
    ]

    /// Groups of visually confusable labels that may share a single spatial cluster
    /// so confidence-weighted voting resolves frame-to-frame jitter.
    nonisolated private static let confusableGroups: [Set<String>] = [
        ["television", "monitor"],
        ["table", "desk"],
        ["cup", "mug"],
        ["chair", "armchair", "optometry chair", "rolling stool"],
        ["cabinet", "filing cabinet", "lens drawer"]
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

    /// Returns `true` if two labels are identical (case-insensitive) or belong to the same
    /// visually confusable category (e.g., `television` / `monitor`, `table` / `desk`).
    nonisolated public static func areLabelsCompatible(_ lhs: String, _ rhs: String) -> Bool {
        let a = lhs.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let b = rhs.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if a == b {
            return true
        }
        return confusableGroups.contains { $0.contains(a) && $0.contains(b) }
    }

    /// Registers a newly projected 3D detection using size-aware clustering,
    /// confidence-weighted multi-frame label voting, and spatial context disambiguation.
    @discardableResult
    public func registerDetection(
        label: String,
        confidence: Float = 1.0,
        worldPosition: simd_float3,
        estimatedWidthMeters: Float? = nil,
        timestamp: Date = Date()
    ) -> TrackedEquipment3D? {
        let trimmedLabel = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLabel.isEmpty else { return nil }
        guard worldPosition.x.isFinite,
              worldPosition.y.isFinite,
              worldPosition.z.isFinite else {
            return nil
        }

        let voteWeight = (confidence.isFinite && confidence > 0) ? confidence : 0.01
        let validWidth: Float? = {
            guard let w = estimatedWidthMeters, w.isFinite, w > 0 else { return nil }
            return w
        }()

        let resolvedLabel = disambiguateLabel(
            trimmedLabel,
            worldPosition: worldPosition,
            estimatedWidthMeters: validWidth
        )

        let affectedID: UUID
        if let matchIndex = findClosestMatchingIndex(label: resolvedLabel, worldPosition: worldPosition) {
            var existing = trackedEquipment[matchIndex]
            let previousCount = Float(max(1, existing.observationCount))
            let newCount = existing.observationCount + 1

            existing.worldPosition = (existing.worldPosition * previousCount + worldPosition) / Float(newCount)
            existing.observationCount = newCount
            existing.lastSeen = max(existing.lastSeen, timestamp)

            if let validWidth {
                if let prevWidth = existing.estimatedWidthMeters {
                    existing.estimatedWidthMeters = (prevWidth * previousCount + validWidth) / Float(newCount)
                } else {
                    existing.estimatedWidthMeters = validWidth
                }
            }

            if existing.labelVotes.isEmpty {
                existing.labelVotes[existing.label.lowercased()] = previousCount * 0.5
            }
            existing.labelVotes[resolvedLabel.lowercased(), default: 0] += voteWeight
            existing.label = Self.winningLabel(from: existing.labelVotes, fallback: resolvedLabel)

            trackedEquipment[matchIndex] = existing
            affectedID = existing.id
        } else {
            let newItem = TrackedEquipment3D(
                id: UUID(),
                label: resolvedLabel,
                worldPosition: worldPosition,
                lastSeen: timestamp,
                observationCount: 1,
                labelVotes: [resolvedLabel.lowercased(): voteWeight],
                estimatedWidthMeters: validWidth
            )
            trackedEquipment.append(newItem)
            affectedID = newItem.id
        }

        reconcileContextDisambiguation()
        return equipment(withID: affectedID)
    }

    /// Replaces the current equipment roster (e.g., when opening a saved scan).
    public func loadEquipment(_ items: [TrackedEquipment3D]) {
        trackedEquipment = items
    }

    /// Returns the tracked item for a given ID, if present.
    public func equipment(withID id: UUID) -> TrackedEquipment3D? {
        trackedEquipment.first { $0.id == id }
    }

    /// Clears all tracked equipment records (e.g., when resetting an AR scanning session).
    public func clearAll() {
        trackedEquipment.removeAll()
    }

    // MARK: - Spatial Context & Label Voting

    private func disambiguateLabel(
        _ rawLabel: String,
        worldPosition: simd_float3,
        estimatedWidthMeters: Float?
    ) -> String {
        let normalized = rawLabel.lowercased()

        if normalized == "television" {
            let isSmallDisplay = estimatedWidthMeters.map { $0 < Self.televisionMaxMonitorWidthMeters } ?? false
            let hasDesktopContext = trackedEquipment.contains { item in
                Self.monitorContextLabels.contains(item.label.lowercased())
                    && simd_distance(worldPosition, item.worldPosition) <= Self.monitorContextRadiusMeters
            }
            if isSmallDisplay || hasDesktopContext {
                return "monitor"
            }
        }

        if normalized == "table" {
            let hasDeskContextAbove = trackedEquipment.contains { item in
                Self.deskContextLabels.contains(item.label.lowercased())
                    && Self.isContextAboveOrOnSurface(surfacePos: worldPosition, itemPos: item.worldPosition, maxDistance: Self.deskContextRadiusMeters)
            }
            if hasDeskContextAbove {
                return "desk"
            }
        }

        return rawLabel
    }

    private func reconcileContextDisambiguation() {
        for index in trackedEquipment.indices {
            var item = trackedEquipment[index]
            let currentNormalized = item.label.lowercased()

            if currentNormalized == "television" {
                let isSmallDisplay = item.estimatedWidthMeters.map { $0 < Self.televisionMaxMonitorWidthMeters } ?? false
                let hasDesktopContext = trackedEquipment.contains { other in
                    other.id != item.id
                        && Self.monitorContextLabels.contains(other.label.lowercased())
                        && simd_distance(item.worldPosition, other.worldPosition) <= Self.monitorContextRadiusMeters
                }
                if isSmallDisplay || hasDesktopContext {
                    let tvVotes = item.labelVotes.removeValue(forKey: "television") ?? 0
                    item.labelVotes["monitor", default: 0] += max(tvVotes, 0.5)
                    item.label = Self.winningLabel(from: item.labelVotes, fallback: "monitor")
                    trackedEquipment[index] = item
                }
            } else if currentNormalized == "table" {
                let hasDeskContextAbove = trackedEquipment.contains { other in
                    other.id != item.id
                        && Self.deskContextLabels.contains(other.label.lowercased())
                        && Self.isContextAboveOrOnSurface(
                            surfacePos: item.worldPosition,
                            itemPos: other.worldPosition,
                            maxDistance: Self.deskContextRadiusMeters
                        )
                }
                if hasDeskContextAbove {
                    let tableVotes = item.labelVotes.removeValue(forKey: "table") ?? 0
                    item.labelVotes["desk", default: 0] += max(tableVotes, 0.5)
                    item.label = Self.winningLabel(from: item.labelVotes, fallback: "desk")
                    trackedEquipment[index] = item
                }
            }
        }
    }

    nonisolated private static func isContextAboveOrOnSurface(
        surfacePos: simd_float3,
        itemPos: simd_float3,
        maxDistance: Float
    ) -> Bool {
        let dist = simd_distance(surfacePos, itemPos)
        // Allow small vertical tolerance (-0.10 m) for raycast noise on tabletop hits.
        return dist.isFinite && dist <= maxDistance && itemPos.y >= (surfacePos.y - 0.10)
    }

    nonisolated private static func winningLabel(from votes: [String: Float], fallback: String) -> String {
        guard let best = votes.max(by: { lhs, rhs in
            if lhs.value != rhs.value {
                return lhs.value < rhs.value
            }
            return lhs.key > rhs.key
        }) else {
            return fallback
        }
        return best.key
    }

    private func findClosestMatchingIndex(label: String, worldPosition: simd_float3) -> Int? {
        var bestIndex: Int?
        var bestDistance = Self.mergeRadius(for: label, fallback: defaultMergeRadius)

        for (index, item) in trackedEquipment.enumerated() {
            guard Self.areLabelsCompatible(item.label, label) else {
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
