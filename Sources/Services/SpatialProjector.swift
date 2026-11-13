import ARKit
import CoreGraphics
import Foundation
import ImageIO
import simd

/// Bridges 2D Vision detections into 3D world-space coordinates via ARKit LiDAR raycasting.
public struct SpatialProjector: Sendable {
    /// Minimum and maximum valid raycast depths (in meters) to reject near-clip noise or distant outliers.
    public let minDepthMeters: Float
    public let maxDepthMeters: Float

    public init(minDepthMeters: Float = 0.15, maxDepthMeters: Float = 15.0) {
        self.minDepthMeters = minDepthMeters
        self.maxDepthMeters = maxDepthMeters
    }

    /// Projects the center of a 2D Vision detection into 3D world space using `.up` orientation.
    public func project(detection: Detection2D, in frame: ARFrame, session: ARSession) -> simd_float3? {
        project(detection: detection, in: frame, session: session, imageOrientation: .up)
    }

    /// Projects the center of a 2D Vision detection into 3D world space, accounting for the
    /// `CGImagePropertyOrientation` passed to `VNImageRequestHandler`.
    ///
    /// Note: `ARRaycastQuery.Target` uses `.existingPlaneGeometry` (the modern replacement for
    /// deprecated `ARHitTestResult.ResultType.existingPlaneUsingGeometry`) and `.estimatedPlane`
    /// (which intersects ARKit's LiDAR scene reconstruction mesh when `sceneReconstruction` is enabled).
    public func project(
        detection: Detection2D,
        in frame: ARFrame,
        session: ARSession,
        imageOrientation: CGImagePropertyOrientation
    ) -> simd_float3? {
        guard let normalizedPoint = Self.normalizedImagePoint(
            fromVisionBox: detection.boundingBox,
            orientation: imageOrientation
        ) else {
            return nil
        }

        let cameraPosition = simd_float3(
            frame.camera.transform.columns.3.x,
            frame.camera.transform.columns.3.y,
            frame.camera.transform.columns.3.z
        )

        var candidates: [simd_float3] = []

        // 1. Raycast against bounded detected planes.
        let planeQuery = frame.raycastQuery(
            from: normalizedPoint,
            allowing: .existingPlaneGeometry,
            alignment: .any
        )
        let planeResults = session.raycast(planeQuery)
        candidates.append(contentsOf: planeResults.compactMap(Self.extractValidPosition))

        // 2. Raycast against ARKit's LiDAR scene mesh / estimated surfaces.
        let meshQuery = frame.raycastQuery(
            from: normalizedPoint,
            allowing: .estimatedPlane,
            alignment: .any
        )
        let meshResults = session.raycast(meshQuery)
        candidates.append(contentsOf: meshResults.compactMap(Self.extractValidPosition))

        return Self.selectNearestPosition(
            from: candidates,
            cameraPosition: cameraPosition,
            minDepth: minDepthMeters,
            maxDepth: maxDepthMeters
        )
    }

    /// Projects the center of a 2D Vision detection into 3D world space using synchronously
    /// extracted camera pose and intrinsics, avoiding any `ARFrame` retention across async boundaries.
    public func project(
        detection: Detection2D,
        cameraTransform: simd_float4x4,
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        imageOrientation: CGImagePropertyOrientation,
        session: ARSession
    ) -> simd_float3? {
        guard let ray = Self.worldRay(
            fromVisionBox: detection.boundingBox,
            cameraTransform: cameraTransform,
            intrinsics: intrinsics,
            imageResolution: imageResolution,
            imageOrientation: imageOrientation
        ) else {
            return nil
        }

        var candidates: [simd_float3] = []

        // 1. Raycast against bounded detected planes.
        let planeQuery = ARRaycastQuery(
            origin: ray.origin,
            direction: ray.direction,
            allowing: .existingPlaneGeometry,
            alignment: .any
        )
        let planeResults = session.raycast(planeQuery)
        candidates.append(contentsOf: planeResults.compactMap(Self.extractValidPosition))

        // 2. Raycast against ARKit's LiDAR scene mesh / estimated surfaces.
        let meshQuery = ARRaycastQuery(
            origin: ray.origin,
            direction: ray.direction,
            allowing: .estimatedPlane,
            alignment: .any
        )
        let meshResults = session.raycast(meshQuery)
        candidates.append(contentsOf: meshResults.compactMap(Self.extractValidPosition))

        return Self.selectNearestPosition(
            from: candidates,
            cameraPosition: ray.origin,
            minDepth: minDepthMeters,
            maxDepth: maxDepthMeters
        )
    }

    /// Projects a 2D Vision detection into 3D world space and estimates its physical width
    /// in meters at the raycast depth using the oriented camera focal length.
    public func projectWithMetadata(
        detection: Detection2D,
        cameraTransform: simd_float4x4,
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        imageOrientation: CGImagePropertyOrientation,
        session: ARSession,
        timestamp: Date = Date()
    ) -> ProjectedDetection3D? {
        guard let worldPosition = project(
            detection: detection,
            cameraTransform: cameraTransform,
            intrinsics: intrinsics,
            imageResolution: imageResolution,
            imageOrientation: imageOrientation,
            session: session
        ) else {
            return nil
        }

        let cameraOrigin = simd_float3(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )
        let depth = simd_distance(cameraOrigin, worldPosition)
        let estimatedWidth = Self.estimatePhysicalWidthMeters(
            fromVisionBox: detection.boundingBox,
            depthMeters: depth,
            intrinsics: intrinsics,
            imageResolution: imageResolution,
            orientation: imageOrientation
        )

        return ProjectedDetection3D(
            label: detection.label,
            confidence: detection.confidence,
            worldPosition: worldPosition,
            estimatedWidthMeters: estimatedWidth,
            timestamp: timestamp
        )
    }

    /// Estimates the physical horizontal width (in meters) of a Vision bounding box
    /// at a known raycast `depthMeters` using pinhole camera intrinsics.
    public static func estimatePhysicalWidthMeters(
        fromVisionBox boundingBox: CGRect,
        depthMeters: Float,
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        orientation: CGImagePropertyOrientation
    ) -> Float? {
        guard boundingBox.size.width.isFinite,
              boundingBox.size.width > 0,
              depthMeters.isFinite,
              depthMeters > 0 else {
            return nil
        }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let rawWidth = Float(imageResolution.width)
        let rawHeight = Float(imageResolution.height)

        let horizontalSpanPixels: Float
        let horizontalFocalLength: Float
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            horizontalSpanPixels = rawHeight
            horizontalFocalLength = fy
        default:
            horizontalSpanPixels = rawWidth
            horizontalFocalLength = fx
        }

        guard horizontalSpanPixels.isFinite, horizontalSpanPixels > 0,
              horizontalFocalLength.isFinite, horizontalFocalLength > 0 else {
            return nil
        }

        let estimated = Float(boundingBox.width) * depthMeters * (horizontalSpanPixels / horizontalFocalLength)
        return (estimated.isFinite && estimated > 0) ? estimated : nil
    }

    /// Computes a world-space ray `(origin, direction)` from a Vision bounding box and
    /// camera intrinsics/transform without retaining an `ARFrame`.
    public static func worldRay(
        fromVisionBox boundingBox: CGRect,
        cameraTransform: simd_float4x4,
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        imageOrientation: CGImagePropertyOrientation
    ) -> (origin: simd_float3, direction: simd_float3)? {
        guard let normalizedPoint = normalizedImagePoint(
            fromVisionBox: boundingBox,
            orientation: imageOrientation
        ) else {
            return nil
        }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]
        let width = Float(imageResolution.width)
        let height = Float(imageResolution.height)

        guard fx.isFinite, fy.isFinite, fx > 0, fy > 0,
              cx.isFinite, cy.isFinite,
              width.isFinite, height.isFinite, width > 0, height > 0 else {
            return nil
        }

        let px = Float(normalizedPoint.x) * width
        let py = Float(normalizedPoint.y) * height

        // ARKit camera coordinates: +X right, +Y up, -Z forward.
        let dirCam = simd_normalize(simd_float3((px - cx) / fx, -(py - cy) / fy, -1.0))
        let worldDir4 = cameraTransform * SIMD4<Float>(dirCam.x, dirCam.y, dirCam.z, 0.0)
        let dirWorld = simd_normalize(simd_float3(worldDir4.x, worldDir4.y, worldDir4.z))
        let originWorld = simd_float3(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )

        guard originWorld.x.isFinite, originWorld.y.isFinite, originWorld.z.isFinite,
              dirWorld.x.isFinite, dirWorld.y.isFinite, dirWorld.z.isFinite else {
            return nil
        }

        return (origin: originWorld, direction: dirWorld)
    }

    /// Converts a Vision normalized bounding box (bottom-left origin, oriented by `orientation`)
    /// into normalized camera image coordinates (top-left origin, raw `ARFrame.capturedImage` space)
    /// expected by `ARFrame.raycastQuery(from:allowing:alignment:)`.
    public static func normalizedImagePoint(
        fromVisionBox boundingBox: CGRect,
        orientation: CGImagePropertyOrientation
    ) -> CGPoint? {
        guard boundingBox.origin.x.isFinite,
              boundingBox.origin.y.isFinite,
              boundingBox.size.width.isFinite,
              boundingBox.size.height.isFinite,
              boundingBox.width > 0,
              boundingBox.height > 0 else {
            return nil
        }

        let vx = min(max(boundingBox.midX, 0.0), 1.0)
        let vy = min(max(boundingBox.midY, 0.0), 1.0)

        let rawPoint: CGPoint
        switch orientation {
        case .up:
            rawPoint = CGPoint(x: vx, y: 1.0 - vy)
        case .upMirrored:
            rawPoint = CGPoint(x: 1.0 - vx, y: 1.0 - vy)
        case .down:
            rawPoint = CGPoint(x: 1.0 - vx, y: vy)
        case .downMirrored:
            rawPoint = CGPoint(x: vx, y: vy)
        case .right:
            // Portrait: raw sensor rotated 90° clockwise for upright Vision inference.
            rawPoint = CGPoint(x: 1.0 - vy, y: 1.0 - vx)
        case .rightMirrored:
            rawPoint = CGPoint(x: 1.0 - vy, y: vx)
        case .left:
            // Portrait upside-down: raw sensor rotated 90° counter-clockwise.
            rawPoint = CGPoint(x: vy, y: vx)
        case .leftMirrored:
            rawPoint = CGPoint(x: vy, y: 1.0 - vx)
        @unknown default:
            rawPoint = CGPoint(x: vx, y: 1.0 - vy)
        }

        return CGPoint(
            x: min(max(rawPoint.x, 0.0), 1.0),
            y: min(max(rawPoint.y, 0.0), 1.0)
        )
    }

    /// Extracts the world-space translation from an `ARRaycastResult`, rejecting non-finite transforms.
    public static func extractValidPosition(from result: ARRaycastResult) -> simd_float3? {
        let col = result.worldTransform.columns.3
        let pos = simd_float3(col.x, col.y, col.z)
        guard pos.x.isFinite, pos.y.isFinite, pos.z.isFinite else {
            return nil
        }
        return pos
    }

    /// Selects the closest valid candidate position within `[minDepth, maxDepth]` of the camera.
    public static func selectNearestPosition(
        from candidates: [simd_float3],
        cameraPosition: simd_float3,
        minDepth: Float = 0.15,
        maxDepth: Float = 15.0
    ) -> simd_float3? {
        var bestPosition: simd_float3?
        var bestDistance = Float.greatestFiniteMagnitude

        for candidate in candidates {
            guard candidate.x.isFinite, candidate.y.isFinite, candidate.z.isFinite else {
                continue
            }
            let distance = simd_distance(cameraPosition, candidate)
            guard distance.isFinite, distance >= minDepth, distance <= maxDepth else {
                continue
            }
            if distance < bestDistance {
                bestDistance = distance
                bestPosition = candidate
            }
        }

        return bestPosition
    }
}
