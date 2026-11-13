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

    /// Volumetric 3D NMS radius for planar displays, whiteboards, furniture, and large equipment.
    nonisolated public static let planarAndFurnitureNMSRadiusMeters: Float = 0.65
    /// Volumetric 3D NMS radius for small tabletop or handheld items.
    nonisolated public static let smallTabletopNMSRadiusMeters: Float = 0.25
    /// Default minimum number of independent keyframe observations required to survive Bake filtering.
    nonisolated public static let defaultMinimumObservationsForBake: Int = 3
    /// Default minimum mean confidence `[0, 1]` required to survive Bake filtering.
    nonisolated public static let defaultMinimumConfidenceForBake: Float = 0.40

    /// Maximum physical width (meters) below which a detected `television` is reclassified as `monitor`.
    nonisolated public static let televisionMaxMonitorWidthMeters: Float = 0.75
    /// Physical width (meters) above which a standalone wall/stand display strongly favors `television`.
    nonisolated public static let televisionPreferredMinWidthMeters: Float = 0.85
    /// Proximity radius (meters) within which desktop context (`desk`, `keyboard`, `mouse`, `laptop`)
    /// reclassifies a nearby `television` as `monitor`.
    nonisolated public static let monitorContextRadiusMeters: Float = 0.80
    /// Proximity radius (meters) within which `monitor`, `laptop`, or `keyboard` above/on a `table`
    /// reclassifies that `table` as `desk`.
    nonisolated public static let deskContextRadiusMeters: Float = 0.60

    nonisolated private static let smallItemKeywords: [String] = [
        "bottle", "cup", "mug", "mouse", "keyboard", "book",
        "pen", "mousepad", "pipette", "thermometer", "barcode scanner",
        "telephone", "cell phone", "tablet", "sample tray", "gloves box",
        "remote", "scissors", "clock", "vase", "laptop"
    ]

    nonisolated private static let planarScreenKeywords: [String] = [
        "television", "monitor", "whiteboard", "tv"
    ]

    nonisolated private static let largeItemKeywords: [String] = [
        "table", "desk", "chair", "sofa", "couch", "phoropter", "slit lamp",
        "autorefractor", "lens drawer", "optometry chair", "armchair",
        "cabinet", "filing cabinet", "shelf", "refrigerator", "whiteboard",
        "wheelchair", "examination light", "rolling stool", "radiator", "sink",
        "door", "bed", "bench", "television", "monitor"
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
        ["television", "monitor", "whiteboard"],
        ["table", "desk"],
        ["cup", "mug"],
        ["chair", "armchair", "optometry chair", "rolling stool", "couch", "sofa"],
        ["cabinet", "filing cabinet", "lens drawer"]
    ]

    @Published public private(set) var trackedEquipment: [TrackedEquipment3D] = []
    @Published public private(set) var observationCandidates: [ObservationCandidate3D] = []

    public let defaultMergeRadius: Float
    private let maxCandidateBufferCount: Int = 5_000

    public init(mergeRadiusMeters: Float = EquipmentClusteringManager.defaultMergeRadiusMeters) {
        self.defaultMergeRadius = max(0.0, mergeRadiusMeters)
    }

    /// Returns the size-aware clustering radius (in meters) for immediate clustering helpers.
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

    /// Returns the volumetric 3D NMS radius used during post-scan Bake consolidation.
    nonisolated public static func volumetricNMSRadius(for label: String) -> Float {
        let normalized = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if smallItemKeywords.contains(where: { normalized.contains($0) }) {
            return smallTabletopNMSRadiusMeters
        }
        if planarScreenKeywords.contains(where: { normalized.contains($0) })
            || largeItemKeywords.contains(where: { normalized.contains($0) }) {
            return planarAndFurnitureNMSRadiusMeters
        }
        return largeItemMergeRadiusMeters
    }

    nonisolated private static func isSmallTabletopClass(_ label: String) -> Bool {
        let normalized = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return smallItemKeywords.contains(where: { normalized.contains($0) })
    }

    /// Returns `true` if two labels are identical (case-insensitive) or belong to the same
    /// visually confusable category (e.g., `television` / `monitor` / `whiteboard`, `table` / `desk`).
    nonisolated public static func areLabelsCompatible(_ lhs: String, _ rhs: String) -> Bool {
        let a = lhs.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let b = rhs.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if a == b {
            return true
        }
        return confusableGroups.contains { $0.contains(a) && $0.contains(b) }
    }

    /// Records a lightweight spatial observation candidate during live AR scanning
    /// without instantiating 3D labels or running full multi-pass NMS until Bake time.
    @discardableResult
    public func recordObservationCandidate(
        label: String,
        confidence: Float,
        worldPosition: simd_float3,
        boundingBox3DExtent: simd_float3 = simd_float3(0.3, 0.3, 0.2),
        timestamp: Date = Date()
    ) -> ObservationCandidate3D? {
        let trimmedLabel = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedLabel.isEmpty,
              confidence.isFinite,
              confidence > 0,
              worldPosition.x.isFinite,
              worldPosition.y.isFinite,
              worldPosition.z.isFinite,
              boundingBox3DExtent.x.isFinite,
              boundingBox3DExtent.y.isFinite,
              boundingBox3DExtent.z.isFinite else {
            return nil
        }

        let candidate = ObservationCandidate3D(
            position: worldPosition,
            boundingBox3DExtent: boundingBox3DExtent,
            label: trimmedLabel,
            confidence: min(max(confidence, 0.0), 1.0),
            timestamp: timestamp
        )

        if observationCandidates.count >= maxCandidateBufferCount {
            observationCandidates.removeFirst(500)
        }
        observationCandidates.append(candidate)
        return candidate
    }

    /// Executes multi-pass "Post-Scan Bake" 3D volumetric NMS, noise filtering,
    /// confidence-weighted winner-takes-all voting, and 3D extent disambiguation over
    /// all recorded live candidates, emitting strictly ONE canonical pin per physical object.
    @discardableResult
    public func consolidateObservations(
        minimumObservations: Int = defaultMinimumObservationsForBake,
        minimumAverageConfidence: Float = defaultMinimumConfidenceForBake
    ) -> [TrackedEquipment3D] {
        struct RawBucket {
            var observations: [ObservationCandidate3D]
            var centroid: simd_float3
            var maxNMSRadius: Float
            var hasLargePlanarOrFurniture: Bool
        }

        guard !observationCandidates.isEmpty else {
            return trackedEquipment
        }

        // Seed 3D NMS buckets from highest-confidence detections first.
        let sortedCandidates = observationCandidates.sorted { lhs, rhs in
            if lhs.confidence != rhs.confidence {
                return lhs.confidence > rhs.confidence
            }
            return lhs.timestamp < rhs.timestamp
        }

        var buckets: [RawBucket] = []

        // Pass 1: Class-agnostic volumetric 3D NMS grouping.
        for candidate in sortedCandidates {
            let candidateRadius = Self.volumetricNMSRadius(for: candidate.label)
            let candidateIsSmall = Self.isSmallTabletopClass(candidate.label)

            var bestIndex: Int?
            var bestDistance = Float.greatestFiniteMagnitude

            for (index, bucket) in buckets.enumerated() {
                let distance = simd_distance(candidate.position, bucket.centroid)
                guard distance.isFinite else { continue }

                // Small tabletop objects (cups/bottles/keyboards) use their tighter 0.25 m threshold
                // against furniture so items sitting on a desk remain distinct, while conflicting
                // same-tier predictions (e.g. television + whiteboard + monitor) merge at 0.65 m.
                let effectiveThreshold: Float
                if candidateIsSmall && bucket.hasLargePlanarOrFurniture {
                    effectiveThreshold = Self.smallTabletopNMSRadiusMeters
                } else if !candidateIsSmall && !bucket.hasLargePlanarOrFurniture {
                    effectiveThreshold = Self.smallTabletopNMSRadiusMeters
                } else {
                    effectiveThreshold = max(bucket.maxNMSRadius, candidateRadius)
                }

                if distance <= effectiveThreshold && distance < bestDistance {
                    bestDistance = distance
                    bestIndex = index
                }
            }

            if let matchIdx = bestIndex {
                var bucket = buckets[matchIdx]
                let n = Float(bucket.observations.count)
                bucket.centroid = (bucket.centroid * n + candidate.position) / (n + 1.0)
                bucket.observations.append(candidate)
                bucket.maxNMSRadius = max(bucket.maxNMSRadius, candidateRadius)
                bucket.hasLargePlanarOrFurniture = bucket.hasLargePlanarOrFurniture || !candidateIsSmall
                buckets[matchIdx] = bucket
            } else {
                buckets.append(
                    RawBucket(
                        observations: [candidate],
                        centroid: candidate.position,
                        maxNMSRadius: candidateRadius,
                        hasLargePlanarOrFurniture: !candidateIsSmall
                    )
                )
            }
        }

        // Pass 2: Minimum observation count and confidence noise gate.
        let minObs = max(1, minimumObservations)
        let survivingBuckets = buckets.filter { bucket in
            guard bucket.observations.count >= minObs else { return false }
            let meanConf = bucket.observations.reduce(Float(0)) { $0 + $1.confidence } / Float(bucket.observations.count)
            return meanConf >= minimumAverageConfidence
        }

        // Pass 3: Winner-takes-all weighted voting & initial physical extent summary.
        struct ProvisionalCluster {
            var id: UUID
            var centroid: simd_float3
            var meanExtent: simd_float3
            var votes: [String: Float]
            var label: String
            var observationCount: Int
            var averageConfidence: Float
            var lastSeen: Date
        }

        var provisional: [ProvisionalCluster] = survivingBuckets.map { bucket in
            var votes: [String: Float] = [:]
            var extentSum = simd_float3.zero
            var confSum: Float = 0
            var latestDate = Date.distantPast

            for obs in bucket.observations {
                let normalized = obs.label.lowercased()
                votes[normalized, default: 0] += max(obs.confidence, 0.01)
                extentSum += obs.boundingBox3DExtent
                confSum += obs.confidence
                if obs.timestamp > latestDate {
                    latestDate = obs.timestamp
                }
            }

            let countFloat = Float(bucket.observations.count)
            let meanExtent = extentSum / countFloat
            let meanConfidence = confSum / countFloat
            let winner = Self.winningLabel(from: votes, fallback: bucket.observations[0].label.lowercased())

            return ProvisionalCluster(
                id: UUID(),
                centroid: bucket.centroid,
                meanExtent: meanExtent,
                votes: votes,
                label: winner,
                observationCount: bucket.observations.count,
                averageConfidence: meanConfidence,
                lastSeen: latestDate
            )
        }

        // Spatial & 3D size heuristics across surviving clusters.
        for i in provisional.indices {
            let current = provisional[i]
            let lower = current.label.lowercased()

            let isScreenOrPlanar = Self.planarScreenKeywords.contains(lower)
                || current.votes.keys.contains(where: { Self.planarScreenKeywords.contains($0) })

            if isScreenOrPlanar && Self.planarScreenKeywords.contains(lower) {
                let hasDesktopContext = provisional.contains { other in
                    other.id != current.id
                        && Self.monitorContextLabels.contains(other.label.lowercased())
                        && simd_distance(current.centroid, other.centroid) <= Self.monitorContextRadiusMeters
                }

                let width = current.meanExtent.x
                let totalVotes = max(current.votes.values.reduce(0, +), 0.001)
                let whiteboardVotes = current.votes["whiteboard"] ?? 0
                let tvVotes = (current.votes["television"] ?? 0) + (current.votes["tv"] ?? 0)
                let monitorVotes = current.votes["monitor"] ?? 0

                if hasDesktopContext || width < Self.televisionMaxMonitorWidthMeters {
                    // Narrow screens or desktop workstation screens resolve to monitor.
                    provisional[i].label = "monitor"
                } else if whiteboardVotes / totalVotes >= 0.65 && width > 1.0 && !hasDesktopContext && tvVotes < whiteboardVotes {
                    // Dominant large elevated planar wall surface without desktop gear stays whiteboard.
                    provisional[i].label = "whiteboard"
                } else if width > Self.televisionPreferredMinWidthMeters && tvVotes >= monitorVotes * 0.5 {
                    // Wide display (> 0.85 m) away from a desk strongly favors television.
                    provisional[i].label = "television"
                } else {
                    provisional[i].label = Self.winningLabel(from: current.votes, fallback: lower)
                }
            } else if lower == "table" {
                let hasDeskContextAbove = provisional.contains { other in
                    other.id != current.id
                        && Self.deskContextLabels.contains(other.label.lowercased())
                        && Self.isContextAboveOrOnSurface(
                            surfacePos: current.centroid,
                            itemPos: other.centroid,
                            maxDistance: Self.deskContextRadiusMeters
                        )
                }
                if hasDeskContextAbove {
                    provisional[i].label = "desk"
                }
            }
        }

        // Pass 4: Final single-identity cross-cluster NMS sweep guarantees 1 pin per physical volume.
        provisional.sort { lhs, rhs in
            let lhsScore = Float(lhs.observationCount) * lhs.averageConfidence
            let rhsScore = Float(rhs.observationCount) * rhs.averageConfidence
            return lhsScore > rhsScore
        }

        var finalClusters: [ProvisionalCluster] = []
        for cluster in provisional {
            let duplicateIndex = finalClusters.firstIndex { existing in
                let threshold = Self.volumetricMergeThreshold(
                    labelA: existing.label,
                    labelB: cluster.label
                )
                return simd_distance(existing.centroid, cluster.centroid) <= threshold
            }

            if let existingIdx = duplicateIndex {
                var target = finalClusters[existingIdx]
                let w1 = Float(target.observationCount)
                let w2 = Float(cluster.observationCount)
                let total = w1 + w2
                target.centroid = (target.centroid * w1 + cluster.centroid * w2) / total
                target.meanExtent = (target.meanExtent * w1 + cluster.meanExtent * w2) / total
                target.averageConfidence = (target.averageConfidence * w1 + cluster.averageConfidence * w2) / total
                target.observationCount += cluster.observationCount
                target.lastSeen = max(target.lastSeen, cluster.lastSeen)
                for (k, v) in cluster.votes {
                    target.votes[k, default: 0] += v
                }
                finalClusters[existingIdx] = target
            } else {
                finalClusters.append(cluster)
            }
        }

        let results = finalClusters.map { cluster in
            TrackedEquipment3D(
                id: cluster.id,
                label: cluster.label,
                worldPosition: cluster.centroid,
                lastSeen: cluster.lastSeen,
                observationCount: cluster.observationCount,
                labelVotes: cluster.votes,
                estimatedWidthMeters: cluster.meanExtent.x,
                boundingBox3DExtent: cluster.meanExtent,
                averageConfidence: cluster.averageConfidence
            )
        }

        trackedEquipment = results
        return results
    }

    nonisolated private static func volumetricMergeThreshold(labelA: String, labelB: String) -> Float {
        let smallA = isSmallTabletopClass(labelA)
        let smallB = isSmallTabletopClass(labelB)
        if smallA || smallB {
            return smallTabletopNMSRadiusMeters
        }
        return planarAndFurnitureNMSRadiusMeters
    }

    /// Registers a newly projected 3D detection immediately (used by direct clustering API callers
    /// and real-time compatibility paths) while also recording a candidate observation.
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

        recordObservationCandidate(
            label: trimmedLabel,
            confidence: voteWeight,
            worldPosition: worldPosition,
            boundingBox3DExtent: simd_float3(validWidth ?? 0.3, validWidth ?? 0.3, 0.2),
            timestamp: timestamp
        )

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
            existing.averageConfidence = (existing.averageConfidence * previousCount + min(max(voteWeight, 0), 1)) / Float(newCount)

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
                estimatedWidthMeters: validWidth,
                boundingBox3DExtent: validWidth.map { simd_float3($0, $0, 0.2) },
                averageConfidence: min(max(voteWeight, 0), 1)
            )
            trackedEquipment.append(newItem)
            affectedID = newItem.id
        }

        reconcileContextDisambiguation()
        return equipment(withID: affectedID)
    }

    /// Replaces the current equipment roster (e.g., when opening a saved scan).
    public func loadEquipment(_ items: [TrackedEquipment3D]) {
        observationCandidates.removeAll()
        trackedEquipment = items
    }

    /// Returns the tracked item for a given ID, if present.
    public func equipment(withID id: UUID) -> TrackedEquipment3D? {
        trackedEquipment.first { $0.id == id }
    }

    /// Clears all tracked equipment records and live observation candidates.
    public func clearAll() {
        observationCandidates.removeAll()
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
