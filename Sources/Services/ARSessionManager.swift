import ARKit
import AVFoundation
import Combine
import CoreVideo
import Foundation
import ImageIO
import UIKit
import simd

/// Pure, testable state machine implementing adaptive keyframe throttling
/// to avoid Neural Engine queue saturation.
public struct KeyframeThrottlePolicy: Sendable, Equatable {
    /// Minimum elapsed time (seconds) before emitting the next keyframe when stationary.
    public let timeInterval: TimeInterval
    /// Minimum camera translation (meters) that triggers an immediate keyframe.
    public let translationThreshold: Float

    public private(set) var lastKeyframeTimestamp: TimeInterval?
    public private(set) var lastKeyframePosition: simd_float3?

    public init(timeInterval: TimeInterval = 1.25, translationThreshold: Float = 0.25) {
        self.timeInterval = min(max(timeInterval, 1.0), 1.5)
        self.translationThreshold = max(0.0, translationThreshold)
    }

    /// Evaluates whether the current frame should be dispatched for on-device ML inference.
    /// Returns `false` if inference is already in flight so the Neural Engine queue never saturates.
    public mutating func shouldEmitKeyframe(
        timestamp: TimeInterval,
        cameraPosition: simd_float3,
        isInferenceInFlight: Bool
    ) -> Bool {
        guard !isInferenceInFlight else {
            return false
        }

        guard timestamp.isFinite,
              cameraPosition.x.isFinite,
              cameraPosition.y.isFinite,
              cameraPosition.z.isFinite else {
            return false
        }

        guard let lastTime = lastKeyframeTimestamp,
              let lastPos = lastKeyframePosition else {
            lastKeyframeTimestamp = timestamp
            lastKeyframePosition = cameraPosition
            return true
        }

        // Handle non-monotonic timestamps (e.g., session reset) safely.
        if timestamp < lastTime {
            lastKeyframeTimestamp = timestamp
            lastKeyframePosition = cameraPosition
            return true
        }

        let elapsed = timestamp - lastTime
        let displacement = simd_distance(cameraPosition, lastPos)

        if elapsed >= timeInterval || displacement >= translationThreshold {
            lastKeyframeTimestamp = timestamp
            lastKeyframePosition = cameraPosition
            return true
        }

        return false
    }

    public mutating func reset() {
        lastKeyframeTimestamp = nil
        lastKeyframePosition = nil
    }
}

/// Lifecycle state of the AR scanning session.
public enum ARSessionState: Sendable, Equatable {
    case idle
    case requestingPermissions
    case running
    case paused
    case interrupted
    case failed(VisualMapperError)
}

/// Manages `ARSession` lifecycle, LiDAR capability checks, camera permissions,
/// mesh anchor tracking, and adaptive keyframe dispatch for equipment detection.
@MainActor
public final class ARSessionManager: NSObject, ObservableObject {
    private struct UncheckedSendableBox<Value>: @unchecked Sendable {
        let value: Value
    }

    /// Holds a single in-flight frame and session reference during active keyframe processing.
    /// `isInferenceInFlight` guarantees at most one keyframe is ever retained concurrently.
    private struct UncheckedFrameContext: @unchecked Sendable {
        let frame: ARFrame
        let session: ARSession
    }

    @Published public private(set) var sessionState: ARSessionState = .idle
    public private(set) var currentCameraTransform: simd_float4x4 = matrix_identity_float4x4
    @Published public private(set) var currentCameraPosition: simd_float3 = .zero
    @Published public private(set) var trackingStatusMessage: String = "Ready to scan"
    @Published public private(set) var meshRevision: UInt64 = 0
    @Published public private(set) var isInferenceInFlight: Bool = false
    @Published public private(set) var sessionRunGeneration: UInt64 = 0

    public let session: ARSession
    private var throttlePolicy: KeyframeThrottlePolicy
    private var lastPublishedPosition: simd_float3 = .zero

    private var meshAnchorsByID: [UUID: ARMeshAnchor] = [:]
    private var pendingUpsertedMeshIDs: Set<UUID> = []
    private var pendingRemovedMeshIDs: Set<UUID> = []

    private var detector: (any EquipmentDetecting)?
    private let projector: SpatialProjector
    private weak var clusteringManager: EquipmentClusteringManager?
    private var inferenceTask: Task<Void, Never>?
    private var activeInferenceID: UUID?

    public static var isLiDARSupported: Bool {
        ARWorldTrackingConfiguration.isSupported &&
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
    }

    public init(
        session: ARSession = ARSession(),
        throttlePolicy: KeyframeThrottlePolicy = KeyframeThrottlePolicy(),
        projector: SpatialProjector = SpatialProjector()
    ) {
        self.session = session
        self.throttlePolicy = throttlePolicy
        self.projector = projector
        super.init()
        self.session.delegate = self
        self.session.delegateQueue = .main
    }

    /// Connects the detection, 3D raycast projection, and clustering pipeline.
    public func configurePipeline(
        detector: (any EquipmentDetecting)?,
        clusteringManager: EquipmentClusteringManager
    ) {
        self.detector = detector
        self.clusteringManager = clusteringManager
    }

    /// Builds a LiDAR-enabled `ARWorldTrackingConfiguration`.
    public func makeWorldTrackingConfiguration() -> ARWorldTrackingConfiguration {
        let configuration = ARWorldTrackingConfiguration()
        configuration.sceneReconstruction = .meshWithClassification
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .none
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        return configuration
    }

    /// Verifies LiDAR availability and camera authorization, then starts the `ARSession`.
    public func startSession(resetTracking: Bool = false) async {
        guard Self.isLiDARSupported else {
            sessionState = .failed(.lidarUnsupported)
            trackingStatusMessage = VisualMapperError.lidarUnsupported.localizedDescription
            return
        }

        sessionState = .requestingPermissions
        let authorized = await verifyCameraPermission()
        guard authorized else {
            sessionState = .failed(.cameraPermissionDenied)
            trackingStatusMessage = VisualMapperError.cameraPermissionDenied.localizedDescription
            return
        }

        let configuration = makeWorldTrackingConfiguration()
        let options: ARSession.RunOptions = resetTracking
            ? [.resetTracking, .removeExistingAnchors]
            : []

        if resetTracking {
            clearMeshState()
            throttlePolicy.reset()
        }

        session.run(configuration, options: options)
        sessionRunGeneration &+= 1
        sessionState = .running
        trackingStatusMessage = "Scanning environment…"
    }

    /// Pauses AR capture and cancels any in-flight keyframe detection task.
    public func pauseSession() {
        activeInferenceID = nil
        inferenceTask?.cancel()
        inferenceTask = nil
        isInferenceInFlight = false
        session.pause()
        if case .running = sessionState {
            sessionState = .paused
            trackingStatusMessage = "Scanning paused"
        }
    }

    /// Resets world tracking, clears reconstructed mesh anchors, and clears clustered equipment.
    public func resetSession() async {
        activeInferenceID = nil
        inferenceTask?.cancel()
        inferenceTask = nil
        isInferenceInFlight = false
        clusteringManager?.clearAll()
        await startSession(resetTracking: true)
    }

    /// Drains incremental mesh anchor additions, updates, and removals since the last render sync.
    public func consumePendingMeshChanges() -> (upserted: [ARMeshAnchor], removed: [UUID]) {
        let upserted = pendingUpsertedMeshIDs.compactMap { meshAnchorsByID[$0] }
        let removed = Array(pendingRemovedMeshIDs)
        pendingUpsertedMeshIDs.removeAll()
        pendingRemovedMeshIDs.removeAll()
        return (upserted, removed)
    }

    /// Returns all currently known mesh anchors (useful when initializing a new view).
    public func allMeshAnchors() -> [ARMeshAnchor] {
        Array(meshAnchorsByID.values)
    }

    /// Maps the current `UIInterfaceOrientation` to the `CGImagePropertyOrientation`
    /// required by Vision so the rear camera sensor buffer is evaluated upright.
    ///
    /// Note: `UIInterfaceOrientation.landscapeRight` (home/USB-C on right) corresponds to
    /// `UIDeviceOrientation.landscapeLeft`, which is the native upright orientation (`.up`)
    /// of the iPhone rear camera sensor.
    nonisolated public static func rearCameraVisionOrientation(
        for interfaceOrientation: UIInterfaceOrientation
    ) -> CGImagePropertyOrientation {
        switch interfaceOrientation {
        case .portrait:
            return .right
        case .portraitUpsideDown:
            return .left
        case .landscapeLeft:
            return .down
        case .landscapeRight:
            return .up
        case .unknown:
            return .right
        @unknown default:
            return .right
        }
    }

    // MARK: - Private Helpers

    private func verifyCameraPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func currentInterfaceOrientation() -> UIInterfaceOrientation {
        let windowScene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first(where: { $0.activationState == .foregroundActive })
            ?? UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first

        return windowScene?.effectiveGeometry.interfaceOrientation ?? .portrait
    }

    private func clearMeshState() {
        let existingIDs = Array(meshAnchorsByID.keys)
        meshAnchorsByID.removeAll()
        pendingUpsertedMeshIDs.removeAll()
        pendingRemovedMeshIDs.formUnion(existingIDs)
        meshRevision &+= 1
    }

    private func handleFrameUpdate(_ frame: ARFrame, in session: ARSession) {
        let transform = frame.camera.transform
        currentCameraTransform = transform

        let cameraPos = simd_float3(
            transform.columns.3.x,
            transform.columns.3.y,
            transform.columns.3.z
        )

        // Throttle SwiftUI position broadcasts to reduce unnecessary view diffs at 60/120 Hz.
        if simd_distance(cameraPos, lastPublishedPosition) >= 0.04 {
            lastPublishedPosition = cameraPos
            currentCameraPosition = cameraPos
        }

        guard case .normal = frame.camera.trackingState else {
            return
        }

        guard let detector else { return }

        guard throttlePolicy.shouldEmitKeyframe(
            timestamp: frame.timestamp,
            cameraPosition: cameraPos,
            isInferenceInFlight: isInferenceInFlight
        ) else {
            return
        }

        let imageOrientation = Self.rearCameraVisionOrientation(for: currentInterfaceOrientation())
        let context = UncheckedFrameContext(frame: frame, session: session)
        let inferenceID = UUID()
        activeInferenceID = inferenceID
        isInferenceInFlight = true

        inferenceTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                if self.activeInferenceID == inferenceID {
                    self.activeInferenceID = nil
                    self.isInferenceInFlight = false
                    self.inferenceTask = nil
                }
            }

            do {
                let detections = try await Self.runDetection(
                    detector: detector,
                    context: context,
                    orientation: imageOrientation
                )

                guard !Task.isCancelled else { return }

                let projectionOrientation: CGImagePropertyOrientation = (detector is LocalEquipmentDetector)
                    ? imageOrientation
                    : .up

                let timestamp = Date()
                for detection in detections {
                    if let worldPos = self.projector.project(
                        detection: detection,
                        in: context.frame,
                        session: context.session,
                        imageOrientation: projectionOrientation
                    ) {
                        self.clusteringManager?.registerDetection(
                            label: detection.label,
                            worldPosition: worldPos,
                            timestamp: timestamp
                        )
                    }
                }
            } catch is CancellationError {
                return
            } catch {
                self.trackingStatusMessage = "Detection warning: \(error.localizedDescription)"
            }
        }
    }

    private nonisolated static func runDetection(
        detector: any EquipmentDetecting,
        context: UncheckedFrameContext,
        orientation: CGImagePropertyOrientation
    ) async throws -> [Detection2D] {
        if let localDetector = detector as? LocalEquipmentDetector {
            return try await localDetector.detect(
                pixelBuffer: context.frame.capturedImage,
                orientation: orientation
            )
        }
        return try await detector.detect(pixelBuffer: context.frame.capturedImage)
    }

    private func handleAnchorsAdded(_ anchors: [ARAnchor]) {
        var didModifyMesh = false
        for case let meshAnchor as ARMeshAnchor in anchors {
            meshAnchorsByID[meshAnchor.identifier] = meshAnchor
            pendingUpsertedMeshIDs.insert(meshAnchor.identifier)
            pendingRemovedMeshIDs.remove(meshAnchor.identifier)
            didModifyMesh = true
        }
        if didModifyMesh {
            meshRevision &+= 1
        }
    }

    private func handleAnchorsUpdated(_ anchors: [ARAnchor]) {
        var didModifyMesh = false
        for case let meshAnchor as ARMeshAnchor in anchors {
            meshAnchorsByID[meshAnchor.identifier] = meshAnchor
            pendingUpsertedMeshIDs.insert(meshAnchor.identifier)
            didModifyMesh = true
        }
        if didModifyMesh {
            meshRevision &+= 1
        }
    }

    private func handleAnchorsRemoved(_ anchors: [ARAnchor]) {
        var didModifyMesh = false
        for case let meshAnchor as ARMeshAnchor in anchors {
            meshAnchorsByID.removeValue(forKey: meshAnchor.identifier)
            pendingUpsertedMeshIDs.remove(meshAnchor.identifier)
            pendingRemovedMeshIDs.insert(meshAnchor.identifier)
            didModifyMesh = true
        }
        if didModifyMesh {
            meshRevision &+= 1
        }
    }

    private func handleTrackingStateChange(_ camera: ARCamera) {
        switch camera.trackingState {
        case .notAvailable:
            trackingStatusMessage = "Tracking unavailable"
        case let .limited(reason):
            switch reason {
            case .initializing:
                trackingStatusMessage = "Initializing LiDAR tracking…"
            case .excessiveMotion:
                trackingStatusMessage = "Move device more slowly"
            case .insufficientFeatures:
                trackingStatusMessage = "Point at textured surfaces"
            case .relocalizing:
                trackingStatusMessage = "Relocalizing spatial map…"
            @unknown default:
                trackingStatusMessage = "Limited tracking"
            }
        case .normal:
            trackingStatusMessage = "LiDAR mapping active"
        }
    }
}

// MARK: - ARSessionDelegate

extension ARSessionManager: ARSessionDelegate {
    nonisolated public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let frameBox = UncheckedSendableBox(value: frame)
        let sessionBox = UncheckedSendableBox(value: session)
        MainActor.assumeIsolated {
            handleFrameUpdate(frameBox.value, in: sessionBox.value)
        }
    }

    nonisolated public func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        let anchorsBox = UncheckedSendableBox(value: anchors)
        MainActor.assumeIsolated {
            handleAnchorsAdded(anchorsBox.value)
        }
    }

    nonisolated public func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        let anchorsBox = UncheckedSendableBox(value: anchors)
        MainActor.assumeIsolated {
            handleAnchorsUpdated(anchorsBox.value)
        }
    }

    nonisolated public func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        let anchorsBox = UncheckedSendableBox(value: anchors)
        MainActor.assumeIsolated {
            handleAnchorsRemoved(anchorsBox.value)
        }
    }

    nonisolated public func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let cameraBox = UncheckedSendableBox(value: camera)
        MainActor.assumeIsolated {
            handleTrackingStateChange(cameraBox.value)
        }
    }

    nonisolated public func session(_ session: ARSession, didFailWithError error: Error) {
        let message = error.localizedDescription
        MainActor.assumeIsolated {
            sessionState = .failed(.arSessionFailed(message))
            trackingStatusMessage = message
        }
    }

    nonisolated public func sessionWasInterrupted(_ session: ARSession) {
        MainActor.assumeIsolated {
            sessionState = .interrupted
            trackingStatusMessage = "AR session interrupted"
        }
    }

    nonisolated public func sessionInterruptionEnded(_ session: ARSession) {
        MainActor.assumeIsolated {
            trackingStatusMessage = "Resuming AR session…"
            Task { @MainActor [weak self] in
                await self?.startSession(resetTracking: false)
            }
        }
    }
}
