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

/// Represents a spatially anchored, clustered 3D piece of medical/laboratory equipment.
public struct TrackedEquipment3D: Identifiable, Sendable, Equatable, Hashable {
    public let id: UUID
    public var label: String
    public var worldPosition: simd_float3
    public var lastSeen: Date
    public var observationCount: Int

    public init(
        id: UUID = UUID(),
        label: String,
        worldPosition: simd_float3,
        lastSeen: Date = Date(),
        observationCount: Int = 1
    ) {
        self.id = id
        self.label = label
        self.worldPosition = worldPosition
        self.lastSeen = lastSeen
        self.observationCount = observationCount
    }
}

/// Errors surfaced across AR capture, ML inference, and spatial projection.
public enum VisualMapperError: Error, LocalizedError, Sendable, Equatable {
    case lidarUnsupported
    case cameraPermissionDenied
    case modelNotFound(String)
    case modelLoadFailed(String)
    case visionInferenceFailed(String)
    case arSessionFailed(String)

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
        }
    }
}
