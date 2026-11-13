import CoreGraphics
import Foundation
import simd

/// Operating profile context used across UI terminology and knowledge card resolution.
public enum EnvironmentMode: String, CaseIterable, Codable, Sendable, Identifiable, Hashable {
    case clinicalLab = "Clinical Lab Mode"
    case householdTest = "Household Test Mode"

    public var id: String { rawValue }

    public var shortTitle: String {
        switch self {
        case .clinicalLab:
            return "Clinical Lab"
        case .householdTest:
            return "Household Test"
        }
    }

    public var subtitle: String {
        switch self {
        case .clinicalLab:
            return "Hospital / Optometry simulation guidelines & FHIR/UDI telemetry"
        case .householdTest:
            return "Residential validation mode with spatial & ergonomic asset profiles"
        }
    }
}

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

/// Lightweight 3D detection candidate recorded silently during live AR capture
/// before post-scan volumetric 3D NMS and label consolidation.
public struct ObservationCandidate3D: Identifiable, Sendable, Equatable, Hashable, Codable {
    public let id: UUID
    public let position: simd_float3
    public let boundingBox3DExtent: simd_float3
    public let label: String
    public let confidence: Float
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        position: simd_float3,
        boundingBox3DExtent: simd_float3 = simd_float3(0.3, 0.3, 0.2),
        label: String,
        confidence: Float,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.position = position
        self.boundingBox3DExtent = boundingBox3DExtent
        self.label = label
        self.confidence = confidence
        self.timestamp = timestamp
    }
}

/// Represents a 3D projected detection with confidence and estimated physical dimensions in meters.
public struct ProjectedDetection3D: Sendable, Equatable {
    public let label: String
    public let confidence: Float
    public let worldPosition: simd_float3
    public let estimatedWidthMeters: Float?
    public let boundingBox3DExtent: simd_float3
    public let timestamp: Date

    public init(
        label: String,
        confidence: Float,
        worldPosition: simd_float3,
        estimatedWidthMeters: Float? = nil,
        boundingBox3DExtent: simd_float3? = nil,
        timestamp: Date = Date()
    ) {
        self.label = label
        self.confidence = confidence
        self.worldPosition = worldPosition
        self.estimatedWidthMeters = estimatedWidthMeters
        self.boundingBox3DExtent = boundingBox3DExtent
            ?? simd_float3(estimatedWidthMeters ?? 0.3, estimatedWidthMeters ?? 0.3, 0.2)
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
    /// Running average physical 3D extent `(width, height, depth)` in meters, when available.
    public var boundingBox3DExtent: simd_float3?
    /// Mean detection confidence across merged observations (`[0, 1]`).
    public var averageConfidence: Float

    private enum CodingKeys: String, CodingKey {
        case id
        case label
        case worldPosition
        case lastSeen
        case observationCount
        case labelVotes
        case estimatedWidthMeters
        case boundingBox3DExtent
        case averageConfidence
    }

    public init(
        id: UUID = UUID(),
        label: String,
        worldPosition: simd_float3,
        lastSeen: Date = Date(),
        observationCount: Int = 1,
        labelVotes: [String: Float] = [:],
        estimatedWidthMeters: Float? = nil,
        boundingBox3DExtent: simd_float3? = nil,
        averageConfidence: Float = 0.85
    ) {
        self.id = id
        self.label = label
        self.worldPosition = worldPosition
        self.lastSeen = lastSeen
        self.observationCount = observationCount
        self.labelVotes = labelVotes
        self.estimatedWidthMeters = estimatedWidthMeters
        self.boundingBox3DExtent = boundingBox3DExtent
        self.averageConfidence = averageConfidence
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        label = try container.decode(String.self, forKey: .label)
        worldPosition = try container.decode(simd_float3.self, forKey: .worldPosition)
        lastSeen = try container.decode(Date.self, forKey: .lastSeen)
        observationCount = try container.decode(Int.self, forKey: .observationCount)
        labelVotes = try container.decodeIfPresent([String: Float].self, forKey: .labelVotes) ?? [:]
        estimatedWidthMeters = try container.decodeIfPresent(Float.self, forKey: .estimatedWidthMeters)
        boundingBox3DExtent = try container.decodeIfPresent(simd_float3.self, forKey: .boundingBox3DExtent)
        averageConfidence = try container.decodeIfPresent(Float.self, forKey: .averageConfidence) ?? 0.85
    }
}

/// Architectural surface categories used for Dollhouse 3D PBR shading and ceiling culling.
public enum RoomSurfaceCategory: String, Codable, Sendable, CaseIterable, Hashable {
    case floor
    case wall
    case table
    case seat
    case furniture
    case door
    case window
    case opening
    case ceiling
    case other

    public var displayName: String {
        rawValue.capitalized
    }
}

/// Visual rendering mode in the baked 3D Dollhouse inspector.
public enum DollhouseRenderMode: String, Codable, Sendable, CaseIterable, Identifiable, Hashable {
    case photorealistic = "Photoreal"
    case semanticMesh = "Mesh Build"

    public var id: String { rawValue }
}

/// Downscaled RGB camera keyframe + exact ARKit pose/intrinsics retained during scanning
/// for post-scan photorealistic UV texture projection onto the LiDAR room mesh.
public struct CapturedTextureKeyframe: Sendable, Equatable, Identifiable {
    public let id: UUID
    public let cameraTransform: simd_float4x4
    public let intrinsics: simd_float3x3
    public let imageResolution: CGSize
    public let jpegData: Data
    public let timestamp: TimeInterval
    public let sectorIndex: Int

    public init(
        id: UUID = UUID(),
        cameraTransform: simd_float4x4,
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        jpegData: Data,
        timestamp: TimeInterval,
        sectorIndex: Int
    ) {
        self.id = id
        self.cameraTransform = cameraTransform
        self.intrinsics = intrinsics
        self.imageResolution = imageResolution
        self.jpegData = jpegData
        self.timestamp = timestamp
        self.sectorIndex = sectorIndex
    }
}

/// Real-time 360-degree scanning coverage metrics tracked across angular sectors around the room.
public struct ScanCoverageState: Sendable, Equatable {
    public static let defaultSectorCount: Int = 12

    public let sectorMeshCounts: [Int]
    public let sectorPhotoCaptured: [Bool]
    public let activeHeadingSector: Int
    public let capturedKeyframeCount: Int
    public let maxKeyframeCount: Int
    public let overallCoverageFraction: Double
    public let guidanceHint: String

    public init(
        sectorMeshCounts: [Int] = Array(repeating: 0, count: ScanCoverageState.defaultSectorCount),
        sectorPhotoCaptured: [Bool] = Array(repeating: false, count: ScanCoverageState.defaultSectorCount),
        activeHeadingSector: Int = 0,
        capturedKeyframeCount: Int = 0,
        maxKeyframeCount: Int = 15,
        overallCoverageFraction: Double = 0.0,
        guidanceHint: String = "Rotate slowly 360° around the room to capture mesh & photos"
    ) {
        self.sectorMeshCounts = sectorMeshCounts
        self.sectorPhotoCaptured = sectorPhotoCaptured
        self.activeHeadingSector = activeHeadingSector
        self.capturedKeyframeCount = capturedKeyframeCount
        self.maxKeyframeCount = maxKeyframeCount
        self.overallCoverageFraction = overallCoverageFraction
        self.guidanceHint = guidanceHint
    }
}

/// A baked, world-space submesh grouped by architectural surface category.
public struct BakedSubmesh: Codable, Sendable, Equatable {
    public let category: RoomSurfaceCategory
    public let positions: [SIMD3<Float>]
    public let normals: [SIMD3<Float>]
    public let texCoords: [SIMD2<Float>]
    public let indices: [UInt32]

    private enum CodingKeys: String, CodingKey {
        case category
        case positions
        case normals
        case texCoords
        case indices
    }

    public init(
        category: RoomSurfaceCategory,
        positions: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        texCoords: [SIMD2<Float>] = [],
        indices: [UInt32]
    ) {
        self.category = category
        self.positions = positions
        self.normals = normals
        self.texCoords = texCoords
        self.indices = indices
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        category = try container.decode(RoomSurfaceCategory.self, forKey: .category)
        positions = try container.decode([SIMD3<Float>].self, forKey: .positions)
        normals = try container.decode([SIMD3<Float>].self, forKey: .normals)
        texCoords = try container.decodeIfPresent([SIMD2<Float>].self, forKey: .texCoords) ?? []
        indices = try container.decode([UInt32].self, forKey: .indices)
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
    public var environmentMode: EnvironmentMode
    public var equipment: [TrackedEquipment3D]
    public let boundsMin: SIMD3<Float>
    public let boundsMax: SIMD3<Float>
    public let vertexCount: Int
    public let triangleCount: Int
    public var hasTextureAtlas: Bool

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case createdAt
        case environmentMode
        case equipment
        case boundsMin
        case boundsMax
        case vertexCount
        case triangleCount
        case hasTextureAtlas
    }

    public init(
        id: UUID = UUID(),
        name: String,
        createdAt: Date = Date(),
        environmentMode: EnvironmentMode = .clinicalLab,
        equipment: [TrackedEquipment3D],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>,
        vertexCount: Int,
        triangleCount: Int,
        hasTextureAtlas: Bool = false
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.environmentMode = environmentMode
        self.equipment = equipment
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
        self.hasTextureAtlas = hasTextureAtlas
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        environmentMode = try container.decodeIfPresent(EnvironmentMode.self, forKey: .environmentMode) ?? .clinicalLab
        equipment = try container.decode([TrackedEquipment3D].self, forKey: .equipment)
        boundsMin = try container.decode(SIMD3<Float>.self, forKey: .boundsMin)
        boundsMax = try container.decode(SIMD3<Float>.self, forKey: .boundsMax)
        vertexCount = try container.decode(Int.self, forKey: .vertexCount)
        triangleCount = try container.decode(Int.self, forKey: .triangleCount)
        hasTextureAtlas = try container.decodeIfPresent(Bool.self, forKey: .hasTextureAtlas) ?? false
    }
}

/// Origin of an equipment intelligence profile (offline bundle vs live enterprise/LLM endpoint).
public enum KnowledgeSource: String, Codable, Sendable, Equatable {
    case offlineKnowledgeBase = "Offline Knowledge Base"
    case cloudLiveAPI = "Aunty.cloud Live Intelligence"
}

/// Rich clinical / functional profile surfaced when inspecting a 3D asset in Dollhouse mode,
/// including HL7/FHIR and UDI fields for enterprise hospital EHR/IoT integration.
public struct EquipmentIntelligenceProfile: Identifiable, Codable, Sendable, Equatable {
    public var id: String { canonicalLabel }
    public let canonicalLabel: String
    public let displayName: String
    public let category: String
    public let clinicalOrFunctionalSummary: String
    public let operationalGuidelines: String
    public let maintenanceOrSafetyNotes: String
    public let assetTagID: String
    public let fhirDeviceCode: String
    public let udibaBarcode: String
    public let source: KnowledgeSource

    public init(
        canonicalLabel: String,
        displayName: String,
        category: String,
        clinicalOrFunctionalSummary: String,
        operationalGuidelines: String,
        maintenanceOrSafetyNotes: String,
        assetTagID: String,
        fhirDeviceCode: String,
        udibaBarcode: String,
        source: KnowledgeSource = .offlineKnowledgeBase
    ) {
        self.canonicalLabel = canonicalLabel
        self.displayName = displayName
        self.category = category
        self.clinicalOrFunctionalSummary = clinicalOrFunctionalSummary
        self.operationalGuidelines = operationalGuidelines
        self.maintenanceOrSafetyNotes = maintenanceOrSafetyNotes
        self.assetTagID = assetTagID
        self.fhirDeviceCode = fhirDeviceCode
        self.udibaBarcode = udibaBarcode
        self.source = source
    }
}

/// Full loaded scan bundle ready for Dollhouse 3D inspection.
public struct BakedRoomScan: Identifiable, Sendable, Equatable {
    public var id: UUID { metadata.id }
    public var metadata: SavedRoomScanMetadata
    public let geometry: BakedRoomGeometry
    public let textureAtlasData: Data?

    public init(
        metadata: SavedRoomScanMetadata,
        geometry: BakedRoomGeometry,
        textureAtlasData: Data? = nil
    ) {
        self.metadata = metadata
        self.geometry = geometry
        self.textureAtlasData = textureAtlasData
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
