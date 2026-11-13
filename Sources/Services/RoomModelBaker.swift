import ARKit
import Foundation
import UIKit
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
/// into an architectural Dollhouse 3D model, projects RGB camera keyframes into a `4×4` texture atlas,
/// fills interior floor dropouts, and generates Wavefront `.obj` exports.
public actor RoomModelBaker {
    private struct CategoryBuilder {
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var texCoords: [SIMD2<Float>] = []
        var indices: [UInt32] = []
    }

    private struct PreparedKeyframe {
        let tileIndex: Int
        let cameraPosition: SIMD3<Float>
        let viewMatrix: simd_float4x4
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
        let width: Float
        let height: Float
    }

    nonisolated private static let atlasCols: Int = 4
    nonisolated private static let atlasRows: Int = 4
    nonisolated private static let tileWidth: Int = 512
    nonisolated private static let tileHeight: Int = 384
    nonisolated private static let maxAtlasKeyframeTiles: Int = 15
    nonisolated private static let paletteTileIndex: Int = 15

    public init() {}

    /// Merges captured mesh anchor snapshots into categorized world-space submeshes.
    public func bake(
        snapshots: [MeshAnchorSnapshot],
        progress: (@Sendable (Double, String) -> Void)? = nil
    ) async throws -> BakedRoomGeometry {
        let result = try await bakeWithTextures(
            snapshots: snapshots,
            keyframes: [],
            fillPlanarGaps: false,
            progress: progress
        )
        return result.geometry
    }

    /// Merges captured mesh anchor snapshots with optional camera-projected UV texturing
    /// (`4×4` atlas) and conservative interior floor hole filling.
    public func bakeWithTextures(
        snapshots: [MeshAnchorSnapshot],
        keyframes: [CapturedTextureKeyframe],
        fillPlanarGaps: Bool = true,
        progress: (@Sendable (Double, String) -> Void)? = nil
    ) async throws -> (geometry: BakedRoomGeometry, textureAtlasJPEGData: Data?) {
        guard !snapshots.isEmpty else {
            throw VisualMapperError.meshBakingFailed(
                "No LiDAR mesh geometry has been captured yet. Scan the room slowly before finishing."
            )
        }

        progress?(0.05, "Extracting LiDAR mesh anchors…")

        let usableKeyframes = Array(keyframes.prefix(Self.maxAtlasKeyframeTiles))
        let hasTexturing = !usableKeyframes.isEmpty
        let preparedKeyframes = Self.prepareKeyframes(usableKeyframes)

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

                // Keyed by `(vertexIdx, tileIndex)` so topological sharing is preserved inside
                // each photo tile while preventing UVs from stretching across different atlas tiles.
                var vertexRemaps: [RoomSurfaceCategory: [Int64: UInt32]] = [:]

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

                        let uvResult: (tileIndex: Int, uv0: SIMD2<Float>, uv1: SIMD2<Float>, uv2: SIMD2<Float>)?
                        if hasTexturing {
                            uvResult = Self.projectTriangleUVs(
                                p0: p0,
                                p1: p1,
                                p2: p2,
                                unitFaceNormal: unitFaceNormal,
                                category: category,
                                keyframes: preparedKeyframes
                            )
                        } else {
                            uvResult = nil
                        }

                        let tileKey = uvResult?.tileIndex ?? 0
                        let cornerUVs: [SIMD2<Float>?] = [uvResult?.uv0, uvResult?.uv1, uvResult?.uv2]
                        let cornerIndices = [i0, i1, i2]

                        var builder = builders[category] ?? CategoryBuilder()
                        var remap = vertexRemaps[category] ?? [:]

                        for corner in 0..<3 {
                            let vertexIdx = cornerIndices[corner]
                            let key = (Int64(vertexIdx) << 8) | Int64(tileKey & 0xFF)
                            if let existing = remap[key] {
                                builder.indices.append(existing)
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
                                if let uv = cornerUVs[corner] {
                                    builder.texCoords.append(uv)
                                }
                                builder.indices.append(newIdx)
                                remap[key] = newIdx

                                boundsMin = simd_min(boundsMin, pos)
                                boundsMax = simd_max(boundsMax, pos)
                            }
                        }

                        vertexRemaps[category] = remap
                        builders[category] = builder
                    }
                }
            }

            let fraction = 0.1 + 0.65 * (Double(anchorIdx + 1) / totalSnapshots)
            progress?(fraction, "Baking room surfaces (\(anchorIdx + 1)/\(snapshots.count))…")
            await Task.yield()
        }

        try Task.checkCancellation()

        if fillPlanarGaps, var floorBuilder = builders[.floor], !floorBuilder.positions.isEmpty {
            progress?(0.80, "Completing interior floor gaps…")
            Self.fillInteriorFloorGaps(
                floorBuilder: &floorBuilder,
                includeTexCoords: hasTexturing,
                preparedKeyframes: preparedKeyframes,
                boundsMin: &boundsMin,
                boundsMax: &boundsMax
            )
            builders[.floor] = floorBuilder
        }

        try Task.checkCancellation()
        progress?(0.88, hasTexturing ? "Projecting RGB camera textures…" : "Finalizing unified 3D Dollhouse mesh…")

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
                texCoords: builder.texCoords,
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

        let atlasData: Data? = hasTexturing
            ? Self.buildTextureAtlasJPEG(from: usableKeyframes)
            : nil

        progress?(1.0, "Dollhouse 3D model ready")

        let geometry = BakedRoomGeometry(
            submeshes: submeshes,
            boundsMin: boundsMin,
            boundsMax: boundsMax,
            vertexCount: totalVertices,
            triangleCount: totalTriangles
        )
        return (geometry, atlasData)
    }

    /// Exports a `BakedRoomGeometry` to ASCII Wavefront `.obj` data grouped by architectural category.
    nonisolated public static func buildOBJData(from geometry: BakedRoomGeometry) -> Data {
        var lines: [String] = []
        lines.reserveCapacity(geometry.vertexCount * 2 + geometry.triangleCount + 16)
        lines.append("# VisualMapper3D Baked Dollhouse Export")
        lines.append("o DollhouseRoom")

        var vertexOffset = 1
        var uvOffset = 1
        for submesh in geometry.submeshes where !submesh.indices.isEmpty {
            lines.append("g \(submesh.category.rawValue)")

            for pos in submesh.positions {
                lines.append("v \(pos.x) \(pos.y) \(pos.z)")
            }
            let hasUVs = submesh.texCoords.count == submesh.positions.count
            if hasUVs {
                for uv in submesh.texCoords {
                    lines.append("vt \(uv.x) \(uv.y)")
                }
            }
            for normal in submesh.normals {
                lines.append("vn \(normal.x) \(normal.y) \(normal.z)")
            }

            let triCount = submesh.indices.count / 3
            for t in 0..<triCount {
                let ia = Int(submesh.indices[t * 3])
                let ib = Int(submesh.indices[t * 3 + 1])
                let ic = Int(submesh.indices[t * 3 + 2])
                let va = ia + vertexOffset
                let vb = ib + vertexOffset
                let vc = ic + vertexOffset
                if hasUVs {
                    let ta = ia + uvOffset
                    let tb = ib + uvOffset
                    let tc = ic + uvOffset
                    lines.append("f \(va)/\(ta)/\(va) \(vb)/\(tb)/\(vb) \(vc)/\(tc)/\(vc)")
                } else {
                    lines.append("f \(va)//\(va) \(vb)//\(vb) \(vc)//\(vc)")
                }
            }

            vertexOffset += submesh.positions.count
            if hasUVs {
                uvOffset += submesh.texCoords.count
            }
        }

        let content = lines.joined(separator: "\n") + "\n"
        return Data(content.utf8)
    }

    // MARK: - Photorealistic UV Projection & Floor Completion Helpers

    nonisolated private static func prepareKeyframes(
        _ keyframes: [CapturedTextureKeyframe]
    ) -> [PreparedKeyframe] {
        var result: [PreparedKeyframe] = []
        for (idx, kf) in keyframes.enumerated() where idx < maxAtlasKeyframeTiles {
            let fx = kf.intrinsics[0][0]
            let fy = kf.intrinsics[1][1]
            let cx = kf.intrinsics[2][0]
            let cy = kf.intrinsics[2][1]
            let width = Float(kf.imageResolution.width)
            let height = Float(kf.imageResolution.height)
            guard fx.isFinite, fy.isFinite, fx > 0, fy > 0,
                  cx.isFinite, cy.isFinite,
                  width.isFinite, height.isFinite, width > 0, height > 0 else {
                continue
            }
            let viewMatrix = kf.cameraTransform.inverse
            let camPos = SIMD3<Float>(
                kf.cameraTransform.columns.3.x,
                kf.cameraTransform.columns.3.y,
                kf.cameraTransform.columns.3.z
            )
            result.append(
                PreparedKeyframe(
                    tileIndex: idx,
                    cameraPosition: camPos,
                    viewMatrix: viewMatrix,
                    fx: fx,
                    fy: fy,
                    cx: cx,
                    cy: cy,
                    width: width,
                    height: height
                )
            )
        }
        return result
    }

    nonisolated private static func projectPoint(
        _ worldPoint: SIMD3<Float>,
        into kf: PreparedKeyframe
    ) -> SIMD2<Float>? {
        let cam = kf.viewMatrix * SIMD4<Float>(worldPoint.x, worldPoint.y, worldPoint.z, 1.0)
        // ARKit camera frame looks down -Z.
        let depth = -cam.z
        guard depth.isFinite, depth >= 0.18, depth <= 7.0 else {
            return nil
        }
        let px = kf.fx * (cam.x / depth) + kf.cx
        let py = kf.cy - kf.fy * (cam.y / depth)
        let uRaw = px / kf.width
        let vRaw = py / kf.height
        guard uRaw.isFinite, vRaw.isFinite,
              uRaw >= 0.015, uRaw <= 0.985,
              vRaw >= 0.015, vRaw <= 0.985 else {
            return nil
        }
        return SIMD2<Float>(uRaw, vRaw)
    }

    nonisolated private static func atlasUV(
        rawUV: SIMD2<Float>,
        tileIndex: Int
    ) -> SIMD2<Float> {
        let col = tileIndex % atlasCols
        let row = tileIndex / atlasCols
        let padU: Float = 2.0 / Float(atlasCols * tileWidth)
        let padV: Float = 2.0 / Float(atlasRows * tileHeight)
        let tileUSize = 1.0 / Float(atlasCols)
        let tileVSize = 1.0 / Float(atlasRows)

        let uMin = Float(col) * tileUSize + padU
        let uSpan = tileUSize - 2.0 * padU
        let yTopMin = Float(row) * tileVSize + padV
        let ySpan = tileVSize - 2.0 * padV

        let u = uMin + min(max(rawUV.x, 0.0), 1.0) * uSpan
        let yTopLeft = yTopMin + min(max(rawUV.y, 0.0), 1.0) * ySpan
        // Flip vertical axis for RealityKit / Wavefront OBJ bottom-left UV origin.
        return SIMD2<Float>(u, 1.0 - yTopLeft)
    }

    nonisolated private static func paletteFallbackUV(
        for category: RoomSurfaceCategory
    ) -> SIMD2<Float> {
        let idx = RoomSurfaceCategory.allCases.firstIndex(of: category) ?? (RoomSurfaceCategory.allCases.count - 1)
        let subCol = idx % 4
        let subRow = idx / 4
        let tileCol = paletteTileIndex % atlasCols
        let tileRow = paletteTileIndex / atlasCols
        let tileUSize = 1.0 / Float(atlasCols)
        let tileVSize = 1.0 / Float(atlasRows)

        let xTL = Float(tileCol) * tileUSize + (Float(subCol) + 0.5) * (tileUSize / 4.0)
        let yTL = Float(tileRow) * tileVSize + (Float(subRow) + 0.5) * (tileVSize / 4.0)
        return SIMD2<Float>(xTL, 1.0 - yTL)
    }

    nonisolated private static func projectTriangleUVs(
        p0: SIMD3<Float>,
        p1: SIMD3<Float>,
        p2: SIMD3<Float>,
        unitFaceNormal: SIMD3<Float>,
        category: RoomSurfaceCategory,
        keyframes: [PreparedKeyframe]
    ) -> (tileIndex: Int, uv0: SIMD2<Float>, uv1: SIMD2<Float>, uv2: SIMD2<Float>) {
        var bestScore: Float = -1.0
        var bestResult: (Int, SIMD2<Float>, SIMD2<Float>, SIMD2<Float>)?

        let centroid = (p0 + p1 + p2) * (1.0 / 3.0)

        for kf in keyframes {
            let toCamera = kf.cameraPosition - centroid
            let worldDist = simd_length(toCamera)
            guard worldDist >= 0.20, worldDist <= 6.5 else { continue }

            let facing = simd_dot(unitFaceNormal, toCamera / worldDist)
            guard facing > 0.12 else { continue }

            guard let raw0 = projectPoint(p0, into: kf),
                  let raw1 = projectPoint(p1, into: kf),
                  let raw2 = projectPoint(p2, into: kf) else {
                continue
            }

            let centerRaw = (raw0 + raw1 + raw2) * (1.0 / 3.0)
            let radialOffset = hypot(centerRaw.x - 0.5, centerRaw.y - 0.5)
            let centerBias = max(0.25, 1.0 - radialOffset)
            let distanceBias = 1.0 / (0.6 + worldDist)
            let score = facing * centerBias * distanceBias

            if score > bestScore {
                bestScore = score
                bestResult = (
                    kf.tileIndex,
                    atlasUV(rawUV: raw0, tileIndex: kf.tileIndex),
                    atlasUV(rawUV: raw1, tileIndex: kf.tileIndex),
                    atlasUV(rawUV: raw2, tileIndex: kf.tileIndex)
                )
            }
        }

        if let bestResult {
            return bestResult
        }

        let fallback = paletteFallbackUV(for: category)
        return (paletteTileIndex, fallback, fallback, fallback)
    }

    nonisolated private static func fillInteriorFloorGaps(
        floorBuilder: inout CategoryBuilder,
        includeTexCoords: Bool,
        preparedKeyframes: [PreparedKeyframe],
        boundsMin: inout SIMD3<Float>,
        boundsMax: inout SIMD3<Float>
    ) {
        guard floorBuilder.positions.count >= 12 else { return }

        var minX = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var minZ = Float.greatestFiniteMagnitude
        var maxZ = -Float.greatestFiniteMagnitude
        var ys: [Float] = []
        ys.reserveCapacity(floorBuilder.positions.count)

        for p in floorBuilder.positions {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minZ = min(minZ, p.z)
            maxZ = max(maxZ, p.z)
            ys.append(p.y)
        }

        let spanX = maxX - minX
        let spanZ = maxZ - minZ
        guard spanX >= 0.9, spanZ >= 0.9 else { return }

        ys.sort()
        let medianY = ys[ys.count / 2]

        let cellSize: Float = 0.30
        let cols = min(64, max(3, Int(ceil(spanX / cellSize))))
        let rows = min(64, max(3, Int(ceil(spanZ / cellSize))))
        let dx = spanX / Float(cols)
        let dz = spanZ / Float(rows)

        var occupied = Array(repeating: false, count: rows * cols)
        var rowMinCol = Array(repeating: cols, count: rows)
        var rowMaxCol = Array(repeating: -1, count: rows)
        var colMinRow = Array(repeating: rows, count: cols)
        var colMaxRow = Array(repeating: -1, count: cols)

        let markPoint: (SIMD3<Float>) -> Void = { p in
            let c = min(cols - 1, max(0, Int((p.x - minX) / dx)))
            let r = min(rows - 1, max(0, Int((p.z - minZ) / dz)))
            occupied[r * cols + c] = true
            rowMinCol[r] = min(rowMinCol[r], c)
            rowMaxCol[r] = max(rowMaxCol[r], c)
            colMinRow[c] = min(colMinRow[c], r)
            colMaxRow[c] = max(colMaxRow[c], r)
        }

        for p in floorBuilder.positions {
            markPoint(p)
        }

        // Also sample edges and centroids of existing floor triangles so large flat triangles
        // mark all grid cells they span cleanly.
        let triCount = floorBuilder.indices.count / 3
        for t in 0..<triCount {
            let i0 = Int(floorBuilder.indices[t * 3])
            let i1 = Int(floorBuilder.indices[t * 3 + 1])
            let i2 = Int(floorBuilder.indices[t * 3 + 2])
            guard i0 < floorBuilder.positions.count,
                  i1 < floorBuilder.positions.count,
                  i2 < floorBuilder.positions.count else {
                continue
            }
            let p0 = floorBuilder.positions[i0]
            let p1 = floorBuilder.positions[i1]
            let p2 = floorBuilder.positions[i2]
            markPoint((p0 + p1 + p2) * (1.0 / 3.0))
            for factor: Float in [0.25, 0.5, 0.75] {
                markPoint(p0 + (p1 - p0) * factor)
                markPoint(p1 + (p2 - p1) * factor)
                markPoint(p2 + (p0 - p2) * factor)
            }
        }

        let upNormal = SIMD3<Float>(0, 1, 0)
        for r in 1..<(rows - 1) {
            for c in 1..<(cols - 1) {
                guard !occupied[r * cols + c] else { continue }
                guard rowMinCol[r] < c && c < rowMaxCol[r],
                      colMinRow[c] < r && r < colMaxRow[c] else {
                    continue
                }

                var neighborCount = 0
                if occupied[(r - 1) * cols + c] { neighborCount += 1 }
                if occupied[(r + 1) * cols + c] { neighborCount += 1 }
                if occupied[r * cols + (c - 1)] { neighborCount += 1 }
                if occupied[r * cols + (c + 1)] { neighborCount += 1 }
                guard neighborCount >= 2 else { continue }

                let x0 = minX + Float(c) * dx
                let x1 = x0 + dx
                let z0 = minZ + Float(r) * dz
                let z1 = z0 + dz

                let q0 = SIMD3<Float>(x0, medianY, z0)
                let q1 = SIMD3<Float>(x0, medianY, z1)
                let q2 = SIMD3<Float>(x1, medianY, z1)
                let q3 = SIMD3<Float>(x1, medianY, z0)

                let baseIdx = UInt32(floorBuilder.positions.count)
                floorBuilder.positions.append(contentsOf: [q0, q1, q2, q3])
                floorBuilder.normals.append(contentsOf: [upNormal, upNormal, upNormal, upNormal])

                if includeTexCoords {
                    let quadUVs = projectQuadUVs(
                        q0: q0, q1: q1, q2: q2, q3: q3,
                        unitFaceNormal: upNormal,
                        category: .floor,
                        keyframes: preparedKeyframes
                    )
                    floorBuilder.texCoords.append(contentsOf: quadUVs)
                }

                floorBuilder.indices.append(contentsOf: [
                    baseIdx, baseIdx + 1, baseIdx + 2,
                    baseIdx, baseIdx + 2, baseIdx + 3
                ])

                boundsMin = simd_min(boundsMin, q0)
                boundsMax = simd_max(boundsMax, q2)
            }
        }
    }

    nonisolated private static func projectQuadUVs(
        q0: SIMD3<Float>,
        q1: SIMD3<Float>,
        q2: SIMD3<Float>,
        q3: SIMD3<Float>,
        unitFaceNormal: SIMD3<Float>,
        category: RoomSurfaceCategory,
        keyframes: [PreparedKeyframe]
    ) -> [SIMD2<Float>] {
        let centroid = (q0 + q1 + q2 + q3) * 0.25
        var bestScore: Float = -1.0
        var bestUVs: [SIMD2<Float>]?

        for kf in keyframes {
            let toCamera = kf.cameraPosition - centroid
            let worldDist = simd_length(toCamera)
            guard worldDist >= 0.20, worldDist <= 6.5 else { continue }
            let facing = simd_dot(unitFaceNormal, toCamera / worldDist)
            guard facing > 0.12 else { continue }

            guard let r0 = projectPoint(q0, into: kf),
                  let r1 = projectPoint(q1, into: kf),
                  let r2 = projectPoint(q2, into: kf),
                  let r3 = projectPoint(q3, into: kf) else {
                continue
            }

            let centerRaw = (r0 + r1 + r2 + r3) * 0.25
            let radialOffset = hypot(centerRaw.x - 0.5, centerRaw.y - 0.5)
            let centerBias = max(0.25, 1.0 - radialOffset)
            let distanceBias = 1.0 / (0.6 + worldDist)
            let score = facing * centerBias * distanceBias
            if score > bestScore {
                bestScore = score
                bestUVs = [
                    atlasUV(rawUV: r0, tileIndex: kf.tileIndex),
                    atlasUV(rawUV: r1, tileIndex: kf.tileIndex),
                    atlasUV(rawUV: r2, tileIndex: kf.tileIndex),
                    atlasUV(rawUV: r3, tileIndex: kf.tileIndex)
                ]
            }
        }

        if let bestUVs {
            return bestUVs
        }
        let fallback = paletteFallbackUV(for: category)
        return [fallback, fallback, fallback, fallback]
    }

    nonisolated private static func buildTextureAtlasJPEG(
        from keyframes: [CapturedTextureKeyframe]
    ) -> Data? {
        guard !keyframes.isEmpty else { return nil }
        return autoreleasepool {
            let atlasSize = CGSize(
                width: CGFloat(atlasCols * tileWidth),
                height: CGFloat(atlasRows * tileHeight)
            )
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1.0
            format.opaque = true

            let renderer = UIGraphicsImageRenderer(size: atlasSize, format: format)
            let renderedImage = renderer.image { context in
                let cgContext = context.cgContext
                cgContext.setFillColor(UIColor(red: 0.14, green: 0.17, blue: 0.22, alpha: 1.0).cgColor)
                cgContext.fill(CGRect(origin: .zero, size: atlasSize))

                for (idx, kf) in keyframes.enumerated() where idx < maxAtlasKeyframeTiles {
                    let col = idx % atlasCols
                    let row = idx / atlasCols
                    let tileRect = CGRect(
                        x: CGFloat(col * tileWidth),
                        y: CGFloat(row * tileHeight),
                        width: CGFloat(tileWidth),
                        height: CGFloat(tileHeight)
                    )
                    if let tileImage = UIImage(data: kf.jpegData) {
                        tileImage.draw(in: tileRect)
                    } else {
                        cgContext.setFillColor(UIColor(red: 0.35, green: 0.42, blue: 0.52, alpha: 1.0).cgColor)
                        cgContext.fill(tileRect)
                    }
                }

                // Populate Tile 15 with category fallback color swatches for surfaces
                // outside any camera keyframe frustum.
                let paletteCol = paletteTileIndex % atlasCols
                let paletteRow = paletteTileIndex / atlasCols
                let paletteOriginX = CGFloat(paletteCol * tileWidth)
                let paletteOriginY = CGFloat(paletteRow * tileHeight)
                let subWidth = CGFloat(tileWidth) / 4.0
                let subHeight = CGFloat(tileHeight) / 4.0

                for (catIdx, category) in RoomSurfaceCategory.allCases.enumerated() {
                    let subCol = catIdx % 4
                    let subRow = catIdx / 4
                    let subRect = CGRect(
                        x: paletteOriginX + CGFloat(subCol) * subWidth,
                        y: paletteOriginY + CGFloat(subRow) * subHeight,
                        width: subWidth,
                        height: subHeight
                    )
                    cgContext.setFillColor(fallbackColor(for: category).cgColor)
                    cgContext.fill(subRect)
                }
            }

            return renderedImage.jpegData(compressionQuality: 0.84)
        }
    }

    nonisolated private static func fallbackColor(for category: RoomSurfaceCategory) -> UIColor {
        switch category {
        case .floor:
            return UIColor(red: 0.30, green: 0.34, blue: 0.40, alpha: 1.0)
        case .wall:
            return UIColor(red: 0.85, green: 0.87, blue: 0.90, alpha: 1.0)
        case .table:
            return UIColor(red: 0.72, green: 0.56, blue: 0.39, alpha: 1.0)
        case .seat:
            return UIColor(red: 0.35, green: 0.53, blue: 0.67, alpha: 1.0)
        case .furniture:
            return UIColor(red: 0.64, green: 0.54, blue: 0.44, alpha: 1.0)
        case .door:
            return UIColor(red: 0.53, green: 0.40, blue: 0.30, alpha: 1.0)
        case .window, .opening:
            return UIColor(red: 0.42, green: 0.72, blue: 0.86, alpha: 1.0)
        case .ceiling:
            return UIColor(red: 0.92, green: 0.93, blue: 0.95, alpha: 1.0)
        case .other:
            return UIColor(red: 0.54, green: 0.58, blue: 0.64, alpha: 1.0)
        }
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
