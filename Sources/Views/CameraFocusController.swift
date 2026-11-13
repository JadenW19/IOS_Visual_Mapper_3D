import Combine
import Foundation
import RealityKit
import simd

/// Controls programmatic virtual camera navigation and smooth cubic ease-in-out
/// transitions when focusing on tracked 3D clinic equipment.
@MainActor
public final class CameraFocusController: ObservableObject {
    nonisolated public static let defaultVantageOffsetMeters: Float = 0.6

    @Published public private(set) var focusedItemID: UUID?
    @Published public private(set) var isVirtualCameraActive: Bool = false
    @Published public private(set) var virtualCameraTransform: Transform = .identity
    @Published public private(set) var currentLookAtTarget: simd_float3 = .zero

    private var animationTask: Task<Void, Never>?
    private var activeCameraTransformProvider: @MainActor () -> simd_float4x4

    public init(cameraTransformProvider: @MainActor @escaping () -> simd_float4x4 = { matrix_identity_float4x4 }) {
        self.activeCameraTransformProvider = cameraTransformProvider
    }

    /// Updates the closure that supplies the current live AR camera world transform.
    public func setLiveCameraTransformProvider(_ provider: @MainActor @escaping () -> simd_float4x4) {
        self.activeCameraTransformProvider = provider
    }

    /// Transitions the virtual camera to inspect the specified `TrackedEquipment3D`
    /// from a vantage point 0.6 m back along the line of sight toward the current camera.
    public func focusOnEquipment(_ item: TrackedEquipment3D, duration: TimeInterval = 0.75) {
        guard item.worldPosition.x.isFinite,
              item.worldPosition.y.isFinite,
              item.worldPosition.z.isFinite else {
            return
        }

        animationTask?.cancel()

        let startTransformMatrix: simd_float4x4 = isVirtualCameraActive
            ? virtualCameraTransform.matrix
            : activeCameraTransformProvider()

        let startPosition = Self.extractPosition(from: startTransformMatrix)
        let startLookAt: simd_float3 = isVirtualCameraActive
            ? currentLookAtTarget
            : startPosition + Self.extractForwardDirection(from: startTransformMatrix)

        let targetPosition = Self.computeVantagePosition(
            itemPosition: item.worldPosition,
            currentCameraPosition: startPosition,
            offsetDistance: Self.defaultVantageOffsetMeters
        )
        let targetLookAt = item.worldPosition
        let endTransformMatrix = Self.lookAtTransform(eye: targetPosition, target: targetLookAt)

        focusedItemID = item.id
        isVirtualCameraActive = true
        virtualCameraTransform = Transform(matrix: startTransformMatrix)
        currentLookAtTarget = startLookAt

        guard duration > 0 else {
            virtualCameraTransform = Transform(matrix: endTransformMatrix)
            currentLookAtTarget = targetLookAt
            return
        }

        animationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let clock = ContinuousClock()
            let startInstant = clock.now
            let totalDurationSeconds = max(duration, 0.001)

            while !Task.isCancelled {
                let elapsed = startInstant.duration(to: clock.now)
                let elapsedSeconds = Double(elapsed.components.seconds)
                    + Double(elapsed.components.attoseconds) / 1e18
                let rawProgress = Float(min(max(elapsedSeconds / totalDurationSeconds, 0.0), 1.0))

                let state = Self.interpolateCameraState(
                    from: startTransformMatrix,
                    to: endTransformMatrix,
                    lookAtStart: startLookAt,
                    lookAtEnd: targetLookAt,
                    progress: rawProgress
                )

                self.virtualCameraTransform = Transform(matrix: state.transform)
                self.currentLookAtTarget = state.lookAtTarget

                if rawProgress >= 1.0 {
                    break
                }

                try? await Task.sleep(nanoseconds: 8_333_333) // ~120 Hz frame pacing
            }
        }
    }

    /// Exits equipment focus mode and restores the live AR spatial tracking camera.
    public func exitFocusMode() {
        animationTask?.cancel()
        animationTask = nil
        focusedItemID = nil
        isVirtualCameraActive = false
    }

    // MARK: - Pure Spatial & Easing Math

    /// Cubic ease-in-out timing curve over `[0, 1]`.
    nonisolated public static func cubicEaseInOut(_ t: Float) -> Float {
        let x = min(max(t, 0.0), 1.0)
        if x < 0.5 {
            return 4.0 * x * x * x
        } else {
            let f = -2.0 * x + 2.0
            return 1.0 - (f * f * f) / 2.0
        }
    }

    /// Computes a vantage point `offsetDistance` (default 0.6 m) back from `itemPosition`
    /// along the line of sight toward `currentCameraPosition`.
    nonisolated public static func computeVantagePosition(
        itemPosition: simd_float3,
        currentCameraPosition: simd_float3,
        offsetDistance: Float = defaultVantageOffsetMeters
    ) -> simd_float3 {
        let toCamera = currentCameraPosition - itemPosition
        let distance = simd_length(toCamera)

        let unitDirection: simd_float3
        if distance.isFinite && distance > 1e-4 {
            unitDirection = toCamera / distance
        } else {
            // Degenerate fallback if the camera coincides with the equipment position.
            unitDirection = simd_normalize(simd_float3(0.0, 0.25, 1.0))
        }

        return itemPosition + unitDirection * offsetDistance
    }

    /// Constructs a right-handed RealityKit camera transform at `eye` looking toward `target`
    /// (local `-Z` points toward `target`, `+Y` is up, `+X` is right).
    nonisolated public static func lookAtTransform(
        eye: simd_float3,
        target: simd_float3,
        up: simd_float3 = simd_float3(0, 1, 0)
    ) -> simd_float4x4 {
        let viewVector = eye - target
        let viewLength = simd_length(viewVector)
        let zAxis = (viewLength.isFinite && viewLength > 1e-5)
            ? viewVector / viewLength
            : simd_float3(0, 0, 1)

        var referenceUp = simd_normalize(up)
        if abs(simd_dot(referenceUp, zAxis)) > 0.999 {
            referenceUp = simd_float3(0, 0, 1)
        }

        let xAxis = simd_normalize(simd_cross(referenceUp, zAxis))
        let yAxis = simd_cross(zAxis, xAxis)

        return simd_float4x4(
            SIMD4<Float>(xAxis.x, xAxis.y, xAxis.z, 0),
            SIMD4<Float>(yAxis.x, yAxis.y, yAxis.z, 0),
            SIMD4<Float>(zAxis.x, zAxis.y, zAxis.z, 0),
            SIMD4<Float>(eye.x, eye.y, eye.z, 1)
        )
    }

    /// Smoothly interpolates camera transform (linear translation + spherical quaternion slerp)
    /// and look-at target using `cubicEaseInOut`.
    nonisolated public static func interpolateCameraState(
        from startTransform: simd_float4x4,
        to endTransform: simd_float4x4,
        lookAtStart: simd_float3,
        lookAtEnd: simd_float3,
        progress: Float
    ) -> (transform: simd_float4x4, lookAtTarget: simd_float3) {
        let eased = cubicEaseInOut(progress)

        let startPos = extractPosition(from: startTransform)
        let endPos = extractPosition(from: endTransform)
        let interpPos = simd_mix(startPos, endPos, simd_float3(repeating: eased))

        let startQuat = simd_normalize(simd_quatf(startTransform))
        var endQuat = simd_normalize(simd_quatf(endTransform))
        if simd_dot(startQuat, endQuat) < 0 {
            endQuat = -endQuat
        }
        let interpQuat = simd_slerp(startQuat, endQuat, eased)

        let interpLookAt = simd_mix(lookAtStart, lookAtEnd, simd_float3(repeating: eased))

        let rotationMatrix = simd_float4x4(interpQuat)
        var result = rotationMatrix
        result.columns.3 = SIMD4<Float>(interpPos.x, interpPos.y, interpPos.z, 1.0)
        return (transform: result, lookAtTarget: interpLookAt)
    }

    nonisolated public static func extractPosition(from transform: simd_float4x4) -> simd_float3 {
        simd_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
    }

    nonisolated public static func extractForwardDirection(from transform: simd_float4x4) -> simd_float3 {
        // RealityKit / ARKit camera looks along local -Z.
        let negativeZ = simd_float3(-transform.columns.2.x, -transform.columns.2.y, -transform.columns.2.z)
        let length = simd_length(negativeZ)
        return length > 1e-5 ? negativeZ / length : simd_float3(0, 0, -1)
    }
}
