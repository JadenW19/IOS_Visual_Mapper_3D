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
    @Published public private(set) var isDollhouseOrbitEnabled: Bool = false
    @Published public private(set) var virtualCameraTransform: Transform = .identity
    @Published public private(set) var currentLookAtTarget: simd_float3 = .zero
    @Published public private(set) var orbitPivot: simd_float3 = .zero
    @Published public private(set) var orbitYaw: Float = 0.785398
    @Published public private(set) var orbitPitch: Float = 0.62
    @Published public private(set) var orbitDistance: Float = 4.5

    private var defaultOrbitPivot: simd_float3 = .zero
    private var defaultOrbitDistance: Float = 4.5
    private var animationTask: Task<Void, Never>?
    private var activeCameraTransformProvider: @MainActor () -> simd_float4x4

    nonisolated public static let defaultOrbitYaw: Float = 0.785398
    nonisolated public static let defaultOrbitPitch: Float = 0.62
    nonisolated public static let minOrbitPitch: Float = 0.08
    nonisolated public static let maxOrbitPitch: Float = 1.48
    nonisolated public static let minOrbitDistance: Float = 0.5
    nonisolated public static let maxOrbitDistance: Float = 35.0

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
        animateCamera(
            from: startTransformMatrix,
            to: endTransformMatrix,
            lookAtStart: startLookAt,
            lookAtEnd: targetLookAt,
            duration: duration
        )
    }

    /// Configures Dollhouse 3D orbit bounds around a baked room and positions the virtual camera
    /// at an elevated isometric estate-agent overview.
    public func configureDollhouseOrbit(
        boundsMin: simd_float3,
        boundsMax: simd_float3,
        duration: TimeInterval = 0.0
    ) {
        let center = (boundsMin + boundsMax) * 0.5
        let extentRadius = max(0.75, simd_length(boundsMax - boundsMin) * 0.5)
        let idealDistance = min(
            Self.maxOrbitDistance,
            max(2.0, extentRadius * 2.15)
        )

        defaultOrbitPivot = center
        defaultOrbitDistance = idealDistance

        isDollhouseOrbitEnabled = true
        isVirtualCameraActive = true
        focusedItemID = nil

        resetToDollhouseOrbit(duration: duration)
    }

    /// Smoothly returns to the home Dollhouse orbit framing (or immediately if `duration <= 0`).
    public func resetToDollhouseOrbit(duration: TimeInterval = 0.6) {
        guard isDollhouseOrbitEnabled else { return }

        let startMatrix = virtualCameraTransform.matrix
        let startLookAt = currentLookAtTarget

        orbitPivot = defaultOrbitPivot
        orbitYaw = Self.defaultOrbitYaw
        orbitPitch = Self.defaultOrbitPitch
        orbitDistance = defaultOrbitDistance
        focusedItemID = nil
        isVirtualCameraActive = true

        let targetMatrix = Self.dollhouseCameraTransform(
            pivot: orbitPivot,
            yaw: orbitYaw,
            pitch: orbitPitch,
            distance: orbitDistance
        )

        if duration > 0 {
            animateCamera(
                from: startMatrix,
                to: targetMatrix,
                lookAtStart: startLookAt,
                lookAtEnd: orbitPivot,
                duration: duration
            )
        } else {
            animationTask?.cancel()
            animationTask = nil
            virtualCameraTransform = Transform(matrix: targetMatrix)
            currentLookAtTarget = orbitPivot
        }
    }

    /// Leaves Dollhouse viewer mode completely and restores Live AR camera control semantics.
    public func disableDollhouseOrbit() {
        animationTask?.cancel()
        animationTask = nil
        focusedItemID = nil
        isDollhouseOrbitEnabled = false
        isVirtualCameraActive = false
    }

    /// Orbits the Dollhouse virtual camera by horizontal/vertical angular deltas in radians.
    public func orbitBy(deltaYaw: Float, deltaPitch: Float) {
        guard isDollhouseOrbitEnabled,
              deltaYaw.isFinite,
              deltaPitch.isFinite else {
            return
        }

        if focusedItemID != nil {
            syncOrbitParametersFromCurrentView(pivot: currentLookAtTarget)
        }
        animationTask?.cancel()
        animationTask = nil
        focusedItemID = nil

        orbitYaw += deltaYaw
        orbitPitch = min(max(orbitPitch + deltaPitch, Self.minOrbitPitch), Self.maxOrbitPitch)
        applyOrbitTransform()
    }

    /// Pans the Dollhouse orbit pivot parallel to the camera's screen right and up axes.
    public func panBy(deltaRight: Float, deltaUp: Float) {
        guard isDollhouseOrbitEnabled,
              deltaRight.isFinite,
              deltaUp.isFinite else {
            return
        }

        if focusedItemID != nil {
            syncOrbitParametersFromCurrentView(pivot: currentLookAtTarget)
        }
        animationTask?.cancel()
        animationTask = nil
        focusedItemID = nil

        let matrix = virtualCameraTransform.matrix
        let right = simd_normalize(simd_float3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z))
        let up = simd_normalize(simd_float3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z))
        orbitPivot += right * deltaRight + up * deltaUp
        applyOrbitTransform()
    }

    /// Zooms the Dollhouse camera inward (> 1.0) or outward (< 1.0) toward the current pivot.
    public func zoomBy(scaleFactor: Float) {
        guard isDollhouseOrbitEnabled,
              scaleFactor.isFinite,
              scaleFactor > 0.01 else {
            return
        }

        if focusedItemID != nil {
            syncOrbitParametersFromCurrentView(pivot: currentLookAtTarget)
        }
        animationTask?.cancel()
        animationTask = nil
        focusedItemID = nil

        orbitDistance = min(
            max(orbitDistance / scaleFactor, Self.minOrbitDistance),
            Self.maxOrbitDistance
        )
        applyOrbitTransform()
    }

    /// Exits equipment focus mode: in Dollhouse mode returns to orbit overview;
    /// in Live AR mode restores the live AR spatial tracking camera.
    public func exitFocusMode() {
        if isDollhouseOrbitEnabled {
            resetToDollhouseOrbit(duration: 0.55)
            return
        }
        animationTask?.cancel()
        animationTask = nil
        focusedItemID = nil
        isVirtualCameraActive = false
    }

    private func syncOrbitParametersFromCurrentView(pivot: simd_float3) {
        let eye = Self.extractPosition(from: virtualCameraTransform.matrix)
        let offset = eye - pivot
        let dist = simd_length(offset)
        orbitPivot = pivot
        orbitDistance = min(max(dist, Self.minOrbitDistance), Self.maxOrbitDistance)
        if dist > 1e-4 {
            let clampedY = min(max(offset.y / dist, -0.999), 0.999)
            orbitPitch = min(max(asin(clampedY), Self.minOrbitPitch), Self.maxOrbitPitch)
            orbitYaw = atan2(offset.x, offset.z)
        }
    }

    private func applyOrbitTransform() {
        let matrix = Self.dollhouseCameraTransform(
            pivot: orbitPivot,
            yaw: orbitYaw,
            pitch: orbitPitch,
            distance: orbitDistance
        )
        isVirtualCameraActive = true
        virtualCameraTransform = Transform(matrix: matrix)
        currentLookAtTarget = orbitPivot
    }

    private func animateCamera(
        from startTransformMatrix: simd_float4x4,
        to endTransformMatrix: simd_float4x4,
        lookAtStart: simd_float3,
        lookAtEnd: simd_float3,
        duration: TimeInterval
    ) {
        animationTask?.cancel()
        virtualCameraTransform = Transform(matrix: startTransformMatrix)
        currentLookAtTarget = lookAtStart

        guard duration > 0 else {
            virtualCameraTransform = Transform(matrix: endTransformMatrix)
            currentLookAtTarget = lookAtEnd
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
                    lookAtStart: lookAtStart,
                    lookAtEnd: lookAtEnd,
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

    // MARK: - Pure Spatial & Easing Math

    /// Computes a world-space camera transform orbiting `pivot` at spherical coordinates `(yaw, pitch, distance)`.
    nonisolated public static func dollhouseCameraTransform(
        pivot: simd_float3,
        yaw: Float,
        pitch: Float,
        distance: Float
    ) -> simd_float4x4 {
        let clampedPitch = min(max(pitch, minOrbitPitch), maxOrbitPitch)
        let clampedDistance = min(max(distance, minOrbitDistance), maxOrbitDistance)
        let cosPitch = cos(clampedPitch)
        let eyeOffset = simd_float3(
            clampedDistance * cosPitch * sin(yaw),
            clampedDistance * sin(clampedPitch),
            clampedDistance * cosPitch * cos(yaw)
        )
        return lookAtTransform(eye: pivot + eyeOffset, target: pivot)
    }

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
