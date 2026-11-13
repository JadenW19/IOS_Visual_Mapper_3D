import CoreGraphics
import Foundation
import simd

/// Represents a 2D object detection produced by the on-device Vision / CoreML pipeline.
/// `boundingBox` uses Vision's normalized coordinate space (`[0, 1] x [0, 1]`, bottom-left origin).
public struct Detection2D: Sendable, Equatable, Hashable {
    public let boundingBox: CGRect
    public let label: String
    public let confidence: Float

    public init(boundingBox: CGRect, label: String, confidence: Float) {
        self.boundingBox = boundingBox
        self.label = label
        self.confidence = confidence
    }
}

/// Represents a 3D projected detection with confidence and estimated physical width in meters.
public struct ProjectedDetection3D: Sendable, Equatable {
    public let label: String
    public let confidence: Float
    public let worldPosition: simd_float3
    public let estimatedWidthMeters: Float?
    public let timestamp: Date

    public init(
        label: String,
        confidence: Float,
        worldPosition: simd_float3,
        estimatedWidthMeters: Float? = nil,
        timestamp: Date = Date()
    ) {
        self.label = label
        self.confidence = confidence
        self.worldPosition = worldPosition
        self.estimatedWidthMeters = estimatedWidthMeters
        self.timestamp = timestamp
    }
}

/// Represents a spatially anchored, clustered 3D piece of equipment or furniture.
public struct TrackedEquipment3D: Identifiable, Sendable, Equatable, Hashable, Codable {
    public let id: UUID
    public var label: String
    public var worldPosition: simd_float3
    public var lastSeen: Date
    public var observationCount: Int
    /// Accumulated confidence votes per class label across multiple frames.
    public var labelVotes: [String: Float]
    /// Running average estimated physical bounding width in meters, when available.
    public var estimatedWidthMeters: Float?

    public init(
        id: UUID = UUID(),
        label: String,
        worldPosition: simd_float3,
        lastSeen: Date = Date(),
        observationCount: Int = 1,
        labelVotes: [String: Float] = [:],
        estimatedWidthMeters: Float? = nil
    ) {
        self.id = id
        self.label = label
        self.worldPosition = worldPosition
        self.lastSeen = lastSeen
        self.observationCount = observationCount
        self.labelVotes = labelVotes
        self.estimatedWidthMeters = estimatedWidthMeters
    }
}

/// Architectural surface categories used for Dollhouse 3D shading and ceiling culling.
public enum RoomSurfaceCategory: String, Codable, Sendable, CaseIterable, Hashable {
    case floor
    case wall
    case furniture
    case opening
    case ceiling
    case other

    public var displayName: String {
        rawValue.capitalized
    }
}

/// A baked, world-space submesh grouped by architectural surface category.
public struct BakedSubmesh: Codable, Sendable, Equatable {
    public let category: RoomSurfaceCategory
    public let positions: [SIMD3<Float>]
    public let normals: [SIMD3<Float>]
    public let indices: [UInt32]

    public init(
        category: RoomSurfaceCategory,
        positions: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        indices: [UInt32]
    ) {
        self.category = category
        self.positions = positions
        self.normals = normals
        self.indices = indices
    }

    public var triangleCount: Int {
        indices.count / 3
    }
}

/// Complete baked 3D geometry of a scanned room in world space.
public struct BakedRoomGeometry: Codable, Sendable, Equatable {
    public let submeshes: [BakedSubmesh]
    public let boundsMin: SIMD3<Float>
    public let boundsMax: SIMD3<Float>
    public let vertexCount: Int
    public let triangleCount: Int

    public init(
        submeshes: [BakedSubmesh],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>,
        vertexCount: Int,
        triangleCount: Int
    ) {
        self.submeshes = submeshes
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
    }

    public var center: SIMD3<Float> {
        (boundsMin + boundsMax) * 0.5
    }

    public var boundingRadius: Float {
        max(0.5, simd_length(boundsMax - boundsMin) * 0.5)
    }
}

/// Persistent metadata describing a saved room scan and its detected equipment roster.
public struct SavedRoomScanMetadata: Identifiable, Codable, Sendable, Equatable, Hashable {
    public let id: UUID
    public var name: String
    public let createdAt: Date
    public var equipment: [TrackedEquipment3D]
    public let boundsMin: SIMD3<Float>
    public let boundsMax: SIMD3<Float>
    public let vertexCount: Int
    public let triangleCount: Int

    public init(
        id: UUID = UUID(),
        name: String,
        createdAt: Date = Date(),
        equipment: [TrackedEquipment3D],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>,
        vertexCount: Int,
        triangleCount: Int
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.equipment = equipment
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
    }
}

/// Full loaded scan bundle ready for Dollhouse 3D inspection.
public struct BakedRoomScan: Identifiable, Sendable, Equatable {
    public var id: UUID { metadata.id }
    public var metadata: SavedRoomScanMetadata
    public let geometry: BakedRoomGeometry

    public init(metadata: SavedRoomScanMetadata, geometry: BakedRoomGeometry) {
        self.metadata = metadata
        self.geometry = geometry
    }
}

/// Errors surfaced across AR capture, ML inference, spatial projection, baking, and persistence.
public enum VisualMapperError: Error, LocalizedError, Sendable, Equatable {
    case lidarUnsupported
    case cameraPermissionDenied
    case modelNotFound(String)
    case modelLoadFailed(String)
    case visionInferenceFailed(String)
    case arSessionFailed(String)
    case meshBakingFailed(String)
    case scanStorageFailed(String)

    public var errorDescription: String? {
        switch self {
        case .lidarUnsupported:
            return "LiDAR scene reconstruction is not supported on this device."
        case .cameraPermissionDenied:
            return "Camera access is required for AR room scanning and equipment detection."
        case let .modelNotFound(name):
            return "Unable to locate CoreML model asset '\(name)' in app bundle."
        case let .modelLoadFailed(reason):
            return "Failed to initialize CoreML Vision model: \(reason)"
        case let .visionInferenceFailed(reason):
            return "On-device Vision inference failed: \(reason)"
        case let .arSessionFailed(reason):
            return "AR session encountered an error: \(reason)"
        case let .meshBakingFailed(reason):
            return "Failed to bake 3D room model: \(reason)"
        case let .scanStorageFailed(reason):
            return "Failed to save or load room scan: \(reason)"
        }
    }
}
