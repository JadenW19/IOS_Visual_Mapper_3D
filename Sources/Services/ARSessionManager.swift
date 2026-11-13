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

    public init(timeInterval: TimeInterval = 0.75, translationThreshold: Float = 0.25) {
        self.timeInterval = max(0.25, timeInterval)
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

    /// Holds only the synchronously extracted `CVPixelBuffer` and scalar camera pose/intrinsics
    /// for a single keyframe—never retaining `ARFrame` across async boundaries.
    private struct ExtractedFramePayload: @unchecked Sendable {
        let pixelBuffer: CVPixelBuffer
        let cameraTransform: simd_float4x4
        let intrinsics: simd_float3x3
        let imageResolution: CGSize
        let imageOrientation: CGImagePropertyOrientation
    }

    @Published public private(set) var sessionState: ARSessionState = .idle
    public private(set) var currentCameraTransform: simd_float4x4 = matrix_identity_float4x4
    @Published public private(set) var currentCameraPosition: simd_float3 = .zero
    @Published public private(set) var trackingStatusMessage: String = "Ready to scan"
    @Published public private(set) var trackedMeshAnchorCount: Int = 0
    @Published public private(set) var isInferenceInFlight: Bool = false
    @Published public private(set) var sessionRunGeneration: UInt64 = 0

    /// Maximum `ARMeshAnchor` count retained in RAM before distant anchors are pruned to avoid Jetsam OOM.
    public static let maxRetainedMeshAnchors: Int = 180
    /// Anchor cap applied immediately when iOS dispatches a low-memory warning.
    public static let memoryWarningMeshAnchorCap: Int = 80

    public let session: ARSession
    private var throttlePolicy: KeyframeThrottlePolicy
    private var lastPublishedPosition: simd_float3 = .zero

    private var meshAnchorsByID: [UUID: ARMeshAnchor] = [:]
    private var memoryWarningCancellable: AnyCancellable?

    private var detector: (any EquipmentDetecting)?
    private let projector: SpatialProjector
    private weak var clusteringManager: EquipmentClusteringManager?
    private var inferenceTask: Task<Void, Never>?
    private var activeInferenceID: UUID?
    private var isProcessingFrame: Bool = false

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

        self.memoryWarningCancellable = NotificationCenter.default
            .publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.handleMemoryWarning()
                }
            }
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
        isProcessingFrame = false
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
        isProcessingFrame = false
        isInferenceInFlight = false
        clusteringManager?.clearAll()
        await startSession(resetTracking: true)
    }

    /// Returns all currently known mesh anchors.
    public func allMeshAnchors() -> [ARMeshAnchor] {
        Array(meshAnchorsByID.values)
    }

    /// Inserts or updates a tracked `ARMeshAnchor` identifier/transform mapping and enforces anchor caps.
    /// Useful for testing and internal anchor lifecycle bookkeeping.
    public func registerMeshAnchorForBookkeeping(_ anchor: ARMeshAnchor) {
        meshAnchorsByID[anchor.identifier] = anchor
        pruneStaleOrDistantMeshAnchors(maxAnchors: Self.maxRetainedMeshAnchors)
        syncMeshAnchorCount()
    }

    /// Prunes the farthest `ARMeshAnchor` entries relative to `currentCameraPosition`
    /// whenever the active anchor count exceeds `maxAnchors`.
    @discardableResult
    public func pruneStaleOrDistantMeshAnchors(maxAnchors: Int) -> Int {
        guard maxAnchors >= 0, meshAnchorsByID.count > maxAnchors else {
            return 0
        }

        let entries = meshAnchorsByID.map { (id, anchor) in
            let pos = simd_float3(anchor.transform.columns.3.x, anchor.transform.columns.3.y, anchor.transform.columns.3.z)
            return (id: id, position: pos)
        }
        let idsToKeep = Self.idsToRetain(
            from: entries,
            relativeTo: currentCameraPosition,
            maxRetained: maxAnchors
        )

        var removed = 0
        for id in Array(meshAnchorsByID.keys) where !idsToKeep.contains(id) {
            meshAnchorsByID.removeValue(forKey: id)
            removed += 1
        }
        syncMeshAnchorCount()
        return removed
    }

    /// Pure distance-based anchor retention helper: keeps the closest `maxRetained` anchors
    /// to `referencePosition` so Jetsam pressure during large scans evicts distant chunks first.
    nonisolated public static func idsToRetain(
        from entries: [(id: UUID, position: simd_float3)],
        relativeTo referencePosition: simd_float3,
        maxRetained: Int
    ) -> Set<UUID> {
        guard maxRetained > 0 else { return [] }
        guard entries.count > maxRetained else {
            return Set(entries.map(\.id))
        }
        let sortedByNearest = entries.sorted { lhs, rhs in
            simd_distance_squared(lhs.position, referencePosition) < simd_distance_squared(rhs.position, referencePosition)
        }
        return Set(sortedByNearest.prefix(maxRetained).map(\.id))
    }

    /// Responds to `UIApplication.didReceiveMemoryWarningNotification` by pruning distant mesh anchors.
    public func handleMemoryWarning() {
        let removed = pruneStaleOrDistantMeshAnchors(maxAnchors: Self.memoryWarningMeshAnchorCap)
        if removed > 0 {
            trackingStatusMessage = "Memory optimized (pruned \(removed) distant mesh chunks)"
        }
    }

    /// Captures thread-safe `Sendable` vertex, normal, face, and classification snapshots
    /// synchronously on the main actor inside `autoreleasepool` so baking can run off-main-thread
    /// with minimal peak transient allocations.
    public func extractMeshSnapshots() -> [MeshAnchorSnapshot] {
        var snapshots: [MeshAnchorSnapshot] = []
        snapshots.reserveCapacity(meshAnchorsByID.count)
        for anchor in meshAnchorsByID.values {
            autoreleasepool {
                snapshots.append(MeshAnchorSnapshot.capture(from: anchor))
            }
        }
        return snapshots
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
        meshAnchorsByID.removeAll()
        syncMeshAnchorCount()
    }

    private func syncMeshAnchorCount() {
        let count = meshAnchorsByID.count
        if trackedMeshAnchorCount != count {
            trackedMeshAnchorCount = count
        }
    }

    private func handleFrameUpdate(_ frame: ARFrame, in session: ARSession) {
        autoreleasepool {
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

            guard !isProcessingFrame else {
                return
            }

            guard case .normal = frame.camera.trackingState else {
                return
            }

            guard let detector else { return }

            guard throttlePolicy.shouldEmitKeyframe(
                timestamp: frame.timestamp,
                cameraPosition: cameraPos,
                isInferenceInFlight: isProcessingFrame
            ) else {
                return
            }

            // Synchronously extract pixel buffer and camera pose/intrinsics so `ARFrame`
            // is released immediately when `session(_:didUpdate:)` returns.
            let imageOrientation = Self.rearCameraVisionOrientation(for: currentInterfaceOrientation())
            let payload = ExtractedFramePayload(
                pixelBuffer: frame.capturedImage,
                cameraTransform: transform,
                intrinsics: frame.camera.intrinsics,
                imageResolution: frame.camera.imageResolution,
                imageOrientation: imageOrientation
            )

            let inferenceID = UUID()
            activeInferenceID = inferenceID
            isProcessingFrame = true
            isInferenceInFlight = true

            inferenceTask = Task.detached(priority: .userInitiated) { [weak self, detector] in
                do {
                    let detections = try await Self.runDetection(
                        detector: detector,
                        payload: payload
                    )

                    guard !Task.isCancelled else {
                        await self?.finishInference(id: inferenceID)
                        return
                    }

                    await self?.completeDetection(
                        detections,
                        payload: payload,
                        isLocalDetector: detector is LocalEquipmentDetector,
                        inferenceID: inferenceID
                    )
                } catch is CancellationError {
                    await self?.finishInference(id: inferenceID)
                } catch {
                    await self?.failInference(message: error.localizedDescription, id: inferenceID)
                }
            }
        }
    }

    private func completeDetection(
        _ detections: [Detection2D],
        payload: ExtractedFramePayload,
        isLocalDetector: Bool,
        inferenceID: UUID
    ) {
        defer { finishInference(id: inferenceID) }
        guard activeInferenceID == inferenceID else { return }

        let projectionOrientation: CGImagePropertyOrientation = isLocalDetector
            ? payload.imageOrientation
            : .up

        let timestamp = Date()
        autoreleasepool {
            for detection in detections {
                if let projected = projector.projectWithMetadata(
                    detection: detection,
                    cameraTransform: payload.cameraTransform,
                    intrinsics: payload.intrinsics,
                    imageResolution: payload.imageResolution,
                    imageOrientation: projectionOrientation,
                    session: session,
                    timestamp: timestamp
                ) {
                    // Capture lightweight observation candidates silently during live scan;
                    // heavy 3D text labels and strict NMS deduplication run once during Bake.
                    clusteringManager?.recordObservationCandidate(
                        label: detection.label,
                        confidence: projected.confidence,
                        worldPosition: projected.worldPosition,
                        boundingBox3DExtent: projected.boundingBox3DExtent,
                        timestamp: projected.timestamp
                    )
                }
            }
        }
    }

    private func failInference(message: String, id: UUID) {
        defer { finishInference(id: id) }
        guard activeInferenceID == id else { return }
        trackingStatusMessage = "Detection warning: \(message)"
    }

    private func finishInference(id: UUID) {
        guard activeInferenceID == id else { return }
        activeInferenceID = nil
        isProcessingFrame = false
        isInferenceInFlight = false
        inferenceTask = nil
    }

    private nonisolated static func runDetection(
        detector: any EquipmentDetecting,
        payload: ExtractedFramePayload
    ) async throws -> [Detection2D] {
        if let localDetector = detector as? LocalEquipmentDetector {
            return try await localDetector.detect(
                pixelBuffer: payload.pixelBuffer,
                orientation: payload.imageOrientation
            )
        }
        return try await detector.detect(pixelBuffer: payload.pixelBuffer)
    }

    private func handleAnchorsAdded(_ anchors: [ARAnchor]) {
        autoreleasepool {
            var didModify = false
            for case let meshAnchor as ARMeshAnchor in anchors {
                meshAnchorsByID[meshAnchor.identifier] = meshAnchor
                didModify = true
            }
            if didModify {
                pruneStaleOrDistantMeshAnchors(maxAnchors: Self.maxRetainedMeshAnchors)
                syncMeshAnchorCount()
            }
        }
    }

    private func handleAnchorsUpdated(_ anchors: [ARAnchor]) {
        autoreleasepool {
            var didModify = false
            for case let meshAnchor as ARMeshAnchor in anchors {
                meshAnchorsByID[meshAnchor.identifier] = meshAnchor
                didModify = true
            }
            if didModify {
                syncMeshAnchorCount()
            }
        }
    }

    private func handleAnchorsRemoved(_ anchors: [ARAnchor]) {
        autoreleasepool {
            var didModify = false
            for case let meshAnchor as ARMeshAnchor in anchors {
                if meshAnchorsByID.removeValue(forKey: meshAnchor.identifier) != nil {
                    didModify = true
                }
            }
            if didModify {
                syncMeshAnchorCount()
            }
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
