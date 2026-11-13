import ARKit
import Foundation
import simd

/// Thread-safe, value-typed snapshot of an `ARMeshAnchor`'s geometry buffers captured on the main actor.
public struct MeshAnchorSnapshot: Sendable {
    public let id: UUID
    public let transform: simd_float4x4

    public let vertexData: Data
    public let vertexCount: Int
    public let vertexStride: Int
    public let vertexOffset: Int

    public let normalData: Data
    public let normalCount: Int
    public let normalStride: Int
    public let normalOffset: Int

    public let indexData: Data
    public let faceCount: Int
    public let bytesPerIndex: Int
    public let indexCountPerPrimitive: Int
    public let isTrianglePrimitive: Bool

    public let classificationData: Data?
    public let classificationCount: Int
    public let classificationStride: Int
    public let classificationOffset: Int

    public init(
        id: UUID,
        transform: simd_float4x4,
        vertexData: Data,
        vertexCount: Int,
        vertexStride: Int,
        vertexOffset: Int,
        normalData: Data,
        normalCount: Int,
        normalStride: Int,
        normalOffset: Int,
        indexData: Data,
        faceCount: Int,
        bytesPerIndex: Int,
        indexCountPerPrimitive: Int,
        isTrianglePrimitive: Bool,
        classificationData: Data?,
        classificationCount: Int,
        classificationStride: Int,
        classificationOffset: Int
    ) {
        self.id = id
        self.transform = transform
        self.vertexData = vertexData
        self.vertexCount = vertexCount
        self.vertexStride = vertexStride
        self.vertexOffset = vertexOffset
        self.normalData = normalData
        self.normalCount = normalCount
        self.normalStride = normalStride
        self.normalOffset = normalOffset
        self.indexData = indexData
        self.faceCount = faceCount
        self.bytesPerIndex = bytesPerIndex
        self.indexCountPerPrimitive = indexCountPerPrimitive
        self.isTrianglePrimitive = isTrianglePrimitive
        self.classificationData = classificationData
        self.classificationCount = classificationCount
        self.classificationStride = classificationStride
        self.classificationOffset = classificationOffset
    }

    @MainActor
    public static func capture(from anchor: ARMeshAnchor) -> MeshAnchorSnapshot {
        let geometry = anchor.geometry
        let vertices = geometry.vertices
        let normals = geometry.normals
        let faces = geometry.faces
        let classification = geometry.classification

        let vertexData = Data(
            bytes: vertices.buffer.contents(),
            count: vertices.buffer.length
        )
        let normalData = Data(
            bytes: normals.buffer.contents(),
            count: normals.buffer.length
        )
        let indexData = Data(
            bytes: faces.buffer.contents(),
            count: faces.buffer.length
        )
        let classificationData = classification.map {
            Data(bytes: $0.buffer.contents(), count: $0.buffer.length)
        }

        return MeshAnchorSnapshot(
            id: anchor.identifier,
            transform: anchor.transform,
            vertexData: vertexData,
            vertexCount: vertices.count,
            vertexStride: vertices.stride,
            vertexOffset: vertices.offset,
            normalData: normalData,
            normalCount: normals.count,
            normalStride: normals.stride,
            normalOffset: normals.offset,
            indexData: indexData,
            faceCount: faces.count,
            bytesPerIndex: faces.bytesPerIndex,
            indexCountPerPrimitive: faces.indexCountPerPrimitive,
            isTrianglePrimitive: faces.primitiveType == .triangle,
            classificationData: classificationData,
            classificationCount: classification?.count ?? 0,
            classificationStride: classification?.stride ?? 1,
            classificationOffset: classification?.offset ?? 0
        )
    }
}

/// Off-main-actor geometry baker that merges accumulated LiDAR `ARMeshAnchor` snapshots
/// into an architectural Dollhouse 3D model and generates standard `.obj` exports.
public actor RoomModelBaker {
    private struct CategoryBuilder {
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []
    }

    public init() {}

    /// Merges captured mesh anchor snapshots into categorized world-space submeshes.
    public func bake(
        snapshots: [MeshAnchorSnapshot],
        progress: (@Sendable (Double, String) -> Void)? = nil
    ) async throws -> BakedRoomGeometry {
        guard !snapshots.isEmpty else {
            throw VisualMapperError.meshBakingFailed(
                "No LiDAR mesh geometry has been captured yet. Scan the room slowly before finishing."
            )
        }

        progress?(0.05, "Extracting LiDAR mesh anchors…")

        var builders: [RoomSurfaceCategory: CategoryBuilder] = [:]
        for category in RoomSurfaceCategory.allCases {
            builders[category] = CategoryBuilder()
        }

        var boundsMin = SIMD3<Float>(
            Float.greatestFiniteMagnitude,
            Float.greatestFiniteMagnitude,
            Float.greatestFiniteMagnitude
        )
        var boundsMax = SIMD3<Float>(
            -Float.greatestFiniteMagnitude,
            -Float.greatestFiniteMagnitude,
            -Float.greatestFiniteMagnitude
        )

        let totalSnapshots = Double(snapshots.count)

        for (anchorIdx, snapshot) in snapshots.enumerated() {
            try Task.checkCancellation()

            guard snapshot.isTrianglePrimitive,
                  snapshot.indexCountPerPrimitive == 3,
                  snapshot.faceCount > 0,
                  snapshot.vertexCount > 0,
                  snapshot.bytesPerIndex == 2 || snapshot.bytesPerIndex == 4
            else {
                continue
            }

            autoreleasepool {
                let worldPositions = Self.transformVertices(snapshot: snapshot)
                let worldNormals = Self.transformNormals(snapshot: snapshot)

                // Per-anchor, per-category vertex index remapping preserves topological sharing
                // inside each anchor without duplicating vertices for every adjacent triangle.
                var vertexRemaps: [RoomSurfaceCategory: [Int: UInt32]] = [:]

                let faceCount = snapshot.faceCount
                let bytesPerIndex = snapshot.bytesPerIndex

                snapshot.indexData.withUnsafeBytes { rawIndexBuffer in
                    for faceIndex in 0..<faceCount {
                        let baseByteOffset = faceIndex * 3 * bytesPerIndex
                        guard let i0 = Self.readIndex(from: rawIndexBuffer, byteOffset: baseByteOffset, bytesPerIndex: bytesPerIndex),
                              let i1 = Self.readIndex(from: rawIndexBuffer, byteOffset: baseByteOffset + bytesPerIndex, bytesPerIndex: bytesPerIndex),
                              let i2 = Self.readIndex(from: rawIndexBuffer, byteOffset: baseByteOffset + 2 * bytesPerIndex, bytesPerIndex: bytesPerIndex),
                              i0 < worldPositions.count,
                              i1 < worldPositions.count,
                              i2 < worldPositions.count,
                              i0 != i1, i1 != i2, i0 != i2
                        else {
                            continue
                        }

                        let p0 = worldPositions[i0]
                        let p1 = worldPositions[i1]
                        let p2 = worldPositions[i2]

                        let edge1 = p1 - p0
                        let edge2 = p2 - p0
                        let cross = simd_cross(edge1, edge2)
                        let crossLength = simd_length(cross)
                        let faceWeightedNormal: SIMD3<Float> = crossLength > 1e-6
                            ? cross
                            : SIMD3<Float>(0, 1, 0)
                        let unitFaceNormal: SIMD3<Float> = crossLength > 1e-6
                            ? cross / crossLength
                            : SIMD3<Float>(0, 1, 0)

                        let category = Self.surfaceCategory(
                            snapshot: snapshot,
                            faceIndex: faceIndex
                        )

                        var builder = builders[category] ?? CategoryBuilder()
                        var remap = vertexRemaps[category] ?? [:]

                        for vertexIdx in [i0, i1, i2] {
                            if let existing = remap[vertexIdx] {
                                builder.indices.append(existing)
                                // Accumulate area-weighted normals when sharing vertices across triangles.
                                let intIdx = Int(existing)
                                if vertexIdx >= worldNormals.count {
                                    builder.normals[intIdx] += faceWeightedNormal
                                }
                            } else {
                                let newIdx = UInt32(builder.positions.count)
                                let pos = worldPositions[vertexIdx]
                                let norm = vertexIdx < worldNormals.count
                                    ? worldNormals[vertexIdx]
                                    : unitFaceNormal

                                builder.positions.append(pos)
                                builder.normals.append(norm)
                                builder.indices.append(newIdx)
                                remap[vertexIdx] = newIdx

                                boundsMin = simd_min(boundsMin, pos)
                                boundsMax = simd_max(boundsMax, pos)
                            }
                        }

                        vertexRemaps[category] = remap
                        builders[category] = builder
                    }
                }
            }

            let fraction = 0.1 + 0.78 * (Double(anchorIdx + 1) / totalSnapshots)
            progress?(fraction, "Baking room surfaces (\(anchorIdx + 1)/\(snapshots.count))…")
            await Task.yield()
        }

        try Task.checkCancellation()
        progress?(0.92, "Finalizing unified 3D Dollhouse mesh…")

        var submeshes: [BakedSubmesh] = []
        var totalVertices = 0
        var totalTriangles = 0

        for category in RoomSurfaceCategory.allCases {
            guard let builder = builders[category], !builder.indices.isEmpty else {
                continue
            }
            let normalizedNormals = builder.normals.map { n -> SIMD3<Float> in
                let len = simd_length(n)
                return len > 1e-6 ? n / len : SIMD3<Float>(0, 1, 0)
            }
            let submesh = BakedSubmesh(
                category: category,
                positions: builder.positions,
                normals: normalizedNormals,
                indices: builder.indices
            )
            totalVertices += submesh.positions.count
            totalTriangles += submesh.triangleCount
            submeshes.append(submesh)
        }

        guard totalTriangles > 0 else {
            throw VisualMapperError.meshBakingFailed(
                "Captured LiDAR anchors did not contain any valid surface triangles."
            )
        }

        progress?(1.0, "Dollhouse 3D model ready")

        return BakedRoomGeometry(
            submeshes: submeshes,
            boundsMin: boundsMin,
            boundsMax: boundsMax,
            vertexCount: totalVertices,
            triangleCount: totalTriangles
        )
    }

    /// Exports a `BakedRoomGeometry` to ASCII Wavefront `.obj` data grouped by architectural category.
    nonisolated public static func buildOBJData(from geometry: BakedRoomGeometry) -> Data {
        var lines: [String] = []
        lines.reserveCapacity(geometry.vertexCount * 2 + geometry.triangleCount + 16)
        lines.append("# VisualMapper3D Baked Dollhouse Export")
        lines.append("o DollhouseRoom")

        var vertexOffset = 1
        for submesh in geometry.submeshes where !submesh.indices.isEmpty {
            lines.append("g \(submesh.category.rawValue)")

            for pos in submesh.positions {
                lines.append("v \(pos.x) \(pos.y) \(pos.z)")
            }
            for normal in submesh.normals {
                lines.append("vn \(normal.x) \(normal.y) \(normal.z)")
            }

            let triCount = submesh.indices.count / 3
            for t in 0..<triCount {
                let a = Int(submesh.indices[t * 3]) + vertexOffset
                let b = Int(submesh.indices[t * 3 + 1]) + vertexOffset
                let c = Int(submesh.indices[t * 3 + 2]) + vertexOffset
                lines.append("f \(a)//\(a) \(b)//\(b) \(c)//\(c)")
            }

            vertexOffset += submesh.positions.count
        }

        let content = lines.joined(separator: "\n") + "\n"
        return Data(content.utf8)
    }

    // MARK: - Buffer Decoding Helpers

    nonisolated private static func transformVertices(snapshot: MeshAnchorSnapshot) -> [SIMD3<Float>] {
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(snapshot.vertexCount)

        snapshot.vertexData.withUnsafeBytes { rawBuffer in
            let floatSize = MemoryLayout<Float>.size
            for idx in 0..<snapshot.vertexCount {
                let base = snapshot.vertexOffset + idx * snapshot.vertexStride
                guard base + 3 * floatSize <= rawBuffer.count else { break }

                let x = rawBuffer.loadUnaligned(fromByteOffset: base, as: Float.self)
                let y = rawBuffer.loadUnaligned(fromByteOffset: base + floatSize, as: Float.self)
                let z = rawBuffer.loadUnaligned(fromByteOffset: base + 2 * floatSize, as: Float.self)

                let world4 = snapshot.transform * simd_float4(x, y, z, 1.0)
                result.append(SIMD3<Float>(world4.x, world4.y, world4.z))
            }
        }

        return result
    }

    nonisolated private static func transformNormals(snapshot: MeshAnchorSnapshot) -> [SIMD3<Float>] {
        guard snapshot.normalCount > 0 else { return [] }

        let upper3x3 = simd_float3x3(columns: (
            SIMD3<Float>(snapshot.transform.columns.0.x, snapshot.transform.columns.0.y, snapshot.transform.columns.0.z),
            SIMD3<Float>(snapshot.transform.columns.1.x, snapshot.transform.columns.1.y, snapshot.transform.columns.1.z),
            SIMD3<Float>(snapshot.transform.columns.2.x, snapshot.transform.columns.2.y, snapshot.transform.columns.2.z)
        ))
        let normalMatrix: simd_float3x3 = abs(upper3x3.determinant) > 1e-6
            ? upper3x3.inverse.transpose
            : matrix_identity_float3x3

        var result: [SIMD3<Float>] = []
        result.reserveCapacity(snapshot.normalCount)

        snapshot.normalData.withUnsafeBytes { rawBuffer in
            let floatSize = MemoryLayout<Float>.size
            for idx in 0..<snapshot.normalCount {
                let base = snapshot.normalOffset + idx * snapshot.normalStride
                guard base + 3 * floatSize <= rawBuffer.count else { break }

                let nx = rawBuffer.loadUnaligned(fromByteOffset: base, as: Float.self)
                let ny = rawBuffer.loadUnaligned(fromByteOffset: base + floatSize, as: Float.self)
                let nz = rawBuffer.loadUnaligned(fromByteOffset: base + 2 * floatSize, as: Float.self)

                let transformed = normalMatrix * SIMD3<Float>(nx, ny, nz)
                let length = simd_length(transformed)
                result.append(length > 1e-6 ? transformed / length : SIMD3<Float>(0, 1, 0))
            }
        }

        return result
    }

    nonisolated private static func readIndex(
        from rawBuffer: UnsafeRawBufferPointer,
        byteOffset: Int,
        bytesPerIndex: Int
    ) -> Int? {
        guard byteOffset >= 0, byteOffset + bytesPerIndex <= rawBuffer.count else {
            return nil
        }

        switch bytesPerIndex {
        case 2:
            return Int(rawBuffer.loadUnaligned(fromByteOffset: byteOffset, as: UInt16.self))
        case 4:
            return Int(rawBuffer.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self))
        default:
            return nil
        }
    }

    nonisolated private static func surfaceCategory(
        snapshot: MeshAnchorSnapshot,
        faceIndex: Int
    ) -> RoomSurfaceCategory {
        guard let classificationData = snapshot.classificationData,
              faceIndex < snapshot.classificationCount
        else {
            return .other
        }

        let offset = snapshot.classificationOffset + faceIndex * snapshot.classificationStride
        guard offset >= 0, offset < classificationData.count else {
            return .other
        }

        let rawClassification = classificationData[offset]
        guard let classification = ARMeshClassification(rawValue: Int(rawClassification)) else {
            return .other
        }

        switch classification {
        case .floor:
            return .floor
        case .wall:
            return .wall
        case .table:
            return .table
        case .seat:
            return .seat
        case .window:
            return .window
        case .door:
            return .door
        case .ceiling:
            return .ceiling
        case .none:
            return .other
        @unknown default:
            return .other
        }
    }
}
