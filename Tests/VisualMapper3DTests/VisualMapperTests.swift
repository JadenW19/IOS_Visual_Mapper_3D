import ARKit
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import RealityKit
import UIKit
import Vision
import XCTest
import simd
@testable import VisualMapperCore

final class VisualMapperTests: XCTestCase {

    // MARK: - EquipmentClusteringManager Tests

    @MainActor
    func testClusteringMergesWithinThresholdAndUsesWeightedAverage() {
        let manager = EquipmentClusteringManager()

        let first = manager.registerDetection(
            label: "phoropter",
            worldPosition: simd_float3(1.0, 0.0, -2.0),
            timestamp: Date(timeIntervalSince1970: 100)
        )
        XCTAssertNotNil(first)
        XCTAssertEqual(manager.trackedEquipment.count, 1)
        XCTAssertEqual(first?.observationCount, 1)

        // 0.20 m offset (< 0.35 m), matching label (case-insensitive)
        let second = manager.registerDetection(
            label: "Phoropter",
            worldPosition: simd_float3(1.2, 0.0, -2.0),
            timestamp: Date(timeIntervalSince1970: 102)
        )
        XCTAssertEqual(manager.trackedEquipment.count, 1)
        XCTAssertEqual(second?.observationCount, 2)
        XCTAssertEqual(second?.worldPosition.x ?? 0, 1.1, accuracy: 1e-4)
        XCTAssertEqual(second?.worldPosition.y ?? 0, 0.0, accuracy: 1e-4)
        XCTAssertEqual(second?.worldPosition.z ?? 0, -2.0, accuracy: 1e-4)

        // Third observation weighted average: (1.1 * 2 + 1.4) / 3 = 1.2
        // Distance from 1.1 to 1.4 is 0.30 m (< 0.35 m)
        let third = manager.registerDetection(
            label: "phoropter",
            worldPosition: simd_float3(1.4, 0.0, -2.0),
            timestamp: Date(timeIntervalSince1970: 105)
        )
        XCTAssertEqual(manager.trackedEquipment.count, 1)
        XCTAssertEqual(third?.observationCount, 3)
        XCTAssertEqual(third?.worldPosition.x ?? 0, 1.2, accuracy: 1e-4)
    }

    @MainActor
    func testClusteringCreatesNewRecordAtOrAboveThresholdOrDifferentLabel() {
        let manager = EquipmentClusteringManager()

        manager.registerDetection(label: "slit lamp", worldPosition: simd_float3(0, 0, -1))
        // Exactly 0.45 m away (>= 0.45 m large-item radius must create a new record)
        manager.registerDetection(label: "slit lamp", worldPosition: simd_float3(0.45, 0, -1))
        XCTAssertEqual(manager.trackedEquipment.count, 2)

        // Small desktop items use a tighter 0.20 m radius so adjacent bottles stay distinct
        manager.registerDetection(label: "bottle", worldPosition: simd_float3(0, 0, -1))
        manager.registerDetection(label: "bottle", worldPosition: simd_float3(0.22, 0, -1))
        XCTAssertEqual(manager.trackedEquipment.count, 4)

        XCTAssertEqual(EquipmentClusteringManager.mergeRadius(for: "bottle"), 0.20, accuracy: 1e-6)
        XCTAssertEqual(EquipmentClusteringManager.mergeRadius(for: "cup"), 0.20, accuracy: 1e-6)
        XCTAssertEqual(EquipmentClusteringManager.mergeRadius(for: "desk"), 0.45, accuracy: 1e-6)
        XCTAssertEqual(EquipmentClusteringManager.mergeRadius(for: "phoropter"), 0.45, accuracy: 1e-6)
        XCTAssertEqual(EquipmentClusteringManager.mergeRadius(for: "monitor"), 0.35, accuracy: 1e-6)

        // Different label at same coordinate must create a new record
        manager.registerDetection(label: "autorefractor", worldPosition: simd_float3(0, 0, -1))
        XCTAssertEqual(manager.trackedEquipment.count, 5)
    }

    @MainActor
    func testClusteringRejectsInvalidInputs() {
        let manager = EquipmentClusteringManager()

        XCTAssertNil(manager.registerDetection(label: "   ", worldPosition: simd_float3(0, 0, -1)))
        XCTAssertNil(manager.registerDetection(label: "monitor", worldPosition: simd_float3(.nan, 0, -1)))
        XCTAssertNil(manager.registerDetection(label: "monitor", worldPosition: simd_float3(0, .infinity, -1)))
        XCTAssertTrue(manager.trackedEquipment.isEmpty)
    }

    // MARK: - KeyframeThrottlePolicy Tests

    func testThrottlePolicyTimeAndTranslationBoundaries() {
        var policy = KeyframeThrottlePolicy(timeInterval: 0.75, translationThreshold: 0.25)

        // First frame emits immediately
        XCTAssertTrue(policy.shouldEmitKeyframe(timestamp: 10.0, cameraPosition: .zero, isInferenceInFlight: false))

        // Stationary before 0.75s -> throttled
        XCTAssertFalse(policy.shouldEmitKeyframe(timestamp: 10.5, cameraPosition: simd_float3(0.10, 0, 0), isInferenceInFlight: false))

        // Inference in flight -> always throttled even when past interval and distance
        XCTAssertFalse(policy.shouldEmitKeyframe(timestamp: 15.0, cameraPosition: simd_float3(2.0, 0, 0), isInferenceInFlight: true))

        // Translation >= 0.25m -> emits immediately
        XCTAssertTrue(policy.shouldEmitKeyframe(timestamp: 10.6, cameraPosition: simd_float3(0.25, 0, 0), isInferenceInFlight: false))

        // Elapsed >= 0.75s while stationary -> emits
        XCTAssertTrue(policy.shouldEmitKeyframe(timestamp: 11.35, cameraPosition: simd_float3(0.25, 0, 0), isInferenceInFlight: false))

        // Non-finite input -> rejected
        XCTAssertFalse(policy.shouldEmitKeyframe(timestamp: .nan, cameraPosition: .zero, isInferenceInFlight: false))

        // Reset clears prior timestamp and position -> next frame emits immediately
        policy.reset()
        XCTAssertNil(policy.lastKeyframeTimestamp)
        XCTAssertNil(policy.lastKeyframePosition)
        XCTAssertTrue(policy.shouldEmitKeyframe(timestamp: 12.40, cameraPosition: simd_float3(0.25, 0, 0), isInferenceInFlight: false))
    }

    // MARK: - SpatialProjector Tests

    func testNormalizedImagePointOrientationMappings() {
        // Vision box centered at (0.2, 0.8) (top-left quadrant in Vision's bottom-left origin)
        let box = CGRect(x: 0.1, y: 0.7, width: 0.2, height: 0.2)

        let upPoint = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .up)
        XCTAssertEqual(upPoint?.x ?? -1, 0.2, accuracy: 1e-6)
        XCTAssertEqual(upPoint?.y ?? -1, 0.2, accuracy: 1e-6)

        let rightPoint = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .right)
        XCTAssertEqual(rightPoint?.x ?? -1, 0.2, accuracy: 1e-6)
        XCTAssertEqual(rightPoint?.y ?? -1, 0.8, accuracy: 1e-6)

        let downPoint = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .down)
        XCTAssertEqual(downPoint?.x ?? -1, 0.8, accuracy: 1e-6)
        XCTAssertEqual(downPoint?.y ?? -1, 0.8, accuracy: 1e-6)

        let leftPoint = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .left)
        XCTAssertEqual(leftPoint?.x ?? -1, 0.8, accuracy: 1e-6)
        XCTAssertEqual(leftPoint?.y ?? -1, 0.2, accuracy: 1e-6)

        XCTAssertNil(SpatialProjector.normalizedImagePoint(fromVisionBox: .zero, orientation: .up))
        XCTAssertNil(SpatialProjector.normalizedImagePoint(fromVisionBox: CGRect(x: .nan, y: 0, width: 0.1, height: 0.1), orientation: .up))
    }

    func testSelectNearestPositionFiltersBoundsAndPicksClosest() {
        let camera = simd_float3(0, 0, 0)
        let candidates: [simd_float3] = [
            simd_float3(0, 0, -0.05),  // Too close (< 0.15 m)
            simd_float3(0, 0, -25.0),  // Too far (> 15.0 m)
            simd_float3(.nan, 0, -1),  // Non-finite
            simd_float3(0, 0, -2.5),
            simd_float3(0, 0, -1.2)    // Nearest valid
        ]

        let selected = SpatialProjector.selectNearestPosition(
            from: candidates,
            cameraPosition: camera,
            minDepth: 0.15,
            maxDepth: 15.0
        )
        XCTAssertNotNil(selected)
        XCTAssertEqual(selected?.z ?? 0, -1.2, accuracy: 1e-5)
    }

    // MARK: - CameraFocusController Tests

    func testCubicEaseInOutAndVantageCalculation() {
        XCTAssertEqual(CameraFocusController.cubicEaseInOut(-0.5), 0.0, accuracy: 1e-6)
        XCTAssertEqual(CameraFocusController.cubicEaseInOut(0.0), 0.0, accuracy: 1e-6)
        XCTAssertEqual(CameraFocusController.cubicEaseInOut(0.5), 0.5, accuracy: 1e-6)
        XCTAssertEqual(CameraFocusController.cubicEaseInOut(1.0), 1.0, accuracy: 1e-6)
        XCTAssertEqual(CameraFocusController.cubicEaseInOut(1.5), 1.0, accuracy: 1e-6)

        let itemPos = simd_float3(0, 1, -2)
        let camPos = simd_float3(0, 1, 0)
        let vantage = CameraFocusController.computeVantagePosition(
            itemPosition: itemPos,
            currentCameraPosition: camPos,
            offsetDistance: 0.6
        )
        XCTAssertEqual(simd_distance(vantage, itemPos), 0.6, accuracy: 1e-5)
        XCTAssertEqual(vantage.z, -1.4, accuracy: 1e-5)

        // Coincident camera and item position must not produce NaN
        let degenerateVantage = CameraFocusController.computeVantagePosition(
            itemPosition: itemPos,
            currentCameraPosition: itemPos,
            offsetDistance: 0.6
        )
        XCTAssertTrue(degenerateVantage.x.isFinite && degenerateVantage.y.isFinite && degenerateVantage.z.isFinite)
        XCTAssertEqual(simd_distance(degenerateVantage, itemPos), 0.6, accuracy: 1e-5)
    }

    @MainActor
    func testFocusOnEquipmentImmediateDuration() {
        let controller = CameraFocusController {
            var m = matrix_identity_float4x4
            m.columns.3 = SIMD4<Float>(0, 1.4, 0, 1)
            return m
        }

        let item = TrackedEquipment3D(
            label: "phoropter",
            worldPosition: simd_float3(0, 1.4, -2.0)
        )
        controller.focusOnEquipment(item, duration: 0)

        XCTAssertTrue(controller.isVirtualCameraActive)
        XCTAssertEqual(controller.focusedItemID, item.id)
        XCTAssertEqual(simd_distance(controller.virtualCameraTransform.translation, item.worldPosition), 0.6, accuracy: 1e-4)

        controller.exitFocusMode()
        XCTAssertFalse(controller.isVirtualCameraActive)
        XCTAssertNil(controller.focusedItemID)

        // Non-finite world position must be ignored safely
        let invalidItem = TrackedEquipment3D(
            label: "phoropter",
            worldPosition: simd_float3(.nan, 0, -1)
        )
        controller.focusOnEquipment(invalidItem, duration: 0)
        XCTAssertFalse(controller.isVirtualCameraActive)
    }

    func testShortestArcInterpolationAndRearCameraOrientation() {
        // Orientation mapping must keep landscapeLeft/landscapeRight distinct for rear camera
        XCTAssertEqual(ARSessionManager.rearCameraVisionOrientation(for: .portrait), .right)
        XCTAssertEqual(ARSessionManager.rearCameraVisionOrientation(for: .portraitUpsideDown), .left)
        XCTAssertEqual(ARSessionManager.rearCameraVisionOrientation(for: .landscapeLeft), .down)
        XCTAssertEqual(ARSessionManager.rearCameraVisionOrientation(for: .landscapeRight), .up)
        XCTAssertEqual(ARSessionManager.rearCameraVisionOrientation(for: .unknown), .right)

        // Antipodal start/end transforms representing a small rotation should still interpolate smoothly
        let start = CameraFocusController.lookAtTransform(
            eye: simd_float3(0, 0, 1),
            target: simd_float3(0, 0, 0)
        )
        let end = CameraFocusController.lookAtTransform(
            eye: simd_float3(0.2, 0, 0.98),
            target: simd_float3(0, 0, 0)
        )
        let mid = CameraFocusController.interpolateCameraState(
            from: start,
            to: end,
            lookAtStart: .zero,
            lookAtEnd: .zero,
            progress: 0.5
        )
        let midForward = CameraFocusController.extractForwardDirection(from: mid.transform)
        XCTAssertLessThan(midForward.z, 0, "Shortest-arc slerp should remain facing toward -Z target")
    }

    // MARK: - LocalEquipmentDetector & Formatting Helpers Tests

    func testRectClampingAndMissingModelError() {
        let outOfBounds = CGRect(x: -0.2, y: 0.8, width: 0.6, height: 0.5)
        let clamped = LocalEquipmentDetector.clampNormalizedRect(outOfBounds)
        XCTAssertEqual(clamped.minX, 0.0, accuracy: 1e-6)
        XCTAssertEqual(clamped.minY, 0.8, accuracy: 1e-6)
        XCTAssertEqual(clamped.maxX, 0.4, accuracy: 1e-6)
        XCTAssertEqual(clamped.maxY, 1.0, accuracy: 1e-6)

        XCTAssertThrowsError(try LocalEquipmentDetector(modelURL: URL(fileURLWithPath: "/nonexistent/model.mlmodelc")))
    }

    func testParseObservationsThresholdAndValidation() {
        XCTAssertEqual(LocalEquipmentDetector.defaultConfidenceThreshold, 0.28, accuracy: 1e-6)
        XCTAssertTrue(LocalEquipmentDetector.parseObservations([]).isEmpty)

        let observations: [LocalEquipmentDetector.RawObjectObservation] = [
            // Below 0.28 threshold -> discarded
            .init(
                boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2),
                labels: [.init(identifier: "bottle", confidence: 0.279)]
            ),
            // Exactly at 0.28 threshold -> retained, whitespace trimmed
            .init(
                boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3),
                labels: [.init(identifier: "  cup \n", confidence: 0.28)]
            ),
            // Picks highest finite confidence even if not first, ignores NaN label
            .init(
                boundingBox: CGRect(x: -0.1, y: 0.7, width: 0.5, height: 0.5),
                labels: [
                    .init(identifier: "monitor", confidence: 0.42),
                    .init(identifier: "invalid", confidence: .nan),
                    .init(identifier: "autorefractor", confidence: 0.87)
                ]
            ),
            // Empty/whitespace-only label -> discarded
            .init(
                boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2),
                labels: [.init(identifier: "   ", confidence: 0.95)]
            ),
            // Empty labels -> discarded
            .init(
                boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.2, height: 0.2),
                labels: []
            ),
            // Zero-area after clamping -> discarded
            .init(
                boundingBox: CGRect(x: 1.2, y: 0.2, width: 0.2, height: 0.2),
                labels: [.init(identifier: "trial lens set", confidence: 0.90)]
            )
        ]

        let parsed = LocalEquipmentDetector.parseObservations(raw: observations)
        XCTAssertEqual(parsed.count, 2)

        XCTAssertEqual(parsed[0].label, "cup")
        XCTAssertEqual(parsed[0].confidence, 0.28, accuracy: 1e-6)
        XCTAssertEqual(parsed[0].boundingBox.origin.x, 0.2, accuracy: 1e-6)

        XCTAssertEqual(parsed[1].label, "autorefractor")
        XCTAssertEqual(parsed[1].confidence, 0.87, accuracy: 1e-6)
        XCTAssertEqual(parsed[1].boundingBox.minX, 0.0, accuracy: 1e-6)
        XCTAssertEqual(parsed[1].boundingBox.maxX, 0.4, accuracy: 1e-6)
        XCTAssertEqual(parsed[1].boundingBox.maxY, 1.0, accuracy: 1e-6)

        // Higher explicit threshold filters out the 0.28 item
        let strictParsed = LocalEquipmentDetector.parseObservations(
            raw: observations,
            confidenceThreshold: 0.50
        )
        XCTAssertEqual(strictParsed.map(\.label), ["autorefractor"])

        // Non-finite threshold falls back to default 0.28 threshold
        let fallbackParsed = LocalEquipmentDetector.parseObservations(
            raw: observations,
            confidenceThreshold: .nan
        )
        XCTAssertEqual(fallbackParsed.map(\.label), ["cup", "autorefractor"])
    }

    func testWorldRayProjectionFromIntrinsics() throws {
        let box = CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2) // centered at (0.5, 0.5)
        let intrinsics = simd_float3x3(columns: (
            SIMD3<Float>(1000, 0, 0),
            SIMD3<Float>(0, 1000, 0),
            SIMD3<Float>(960, 720, 1)
        ))
        var cameraTransform = matrix_identity_float4x4
        cameraTransform.columns.3 = SIMD4<Float>(1.0, 1.5, 2.0, 1.0)

        let ray = try XCTUnwrap(SpatialProjector.worldRay(
            fromVisionBox: box,
            cameraTransform: cameraTransform,
            intrinsics: intrinsics,
            imageResolution: CGSize(width: 1920, height: 1440),
            imageOrientation: .up
        ))

        XCTAssertEqual(ray.origin.x, 1.0, accuracy: 1e-5)
        XCTAssertEqual(ray.origin.y, 1.5, accuracy: 1e-5)
        XCTAssertEqual(ray.origin.z, 2.0, accuracy: 1e-5)
        XCTAssertEqual(ray.direction.x, 0.0, accuracy: 1e-5)
        XCTAssertEqual(ray.direction.y, 0.0, accuracy: 1e-5)
        XCTAssertEqual(ray.direction.z, -1.0, accuracy: 1e-5)
    }

    func testBundledCoreMLModelInferenceWhenPresent() async throws {
        let testBundle = Bundle(for: VisualMapperTests.self)
        let modelBundle: Bundle
        if testBundle.url(forResource: "yolov8s-worldv2", withExtension: "mlmodelc") != nil
            || testBundle.url(forResource: "yolov8s-worldv2", withExtension: "mlpackage") != nil {
            modelBundle = testBundle
        } else if Bundle.main.url(forResource: "yolov8s-worldv2", withExtension: "mlmodelc") != nil {
            modelBundle = .main
        } else {
            return
        }

        let detector = try LocalEquipmentDetector(bundle: modelBundle)
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            640,
            640,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        let buffer = try XCTUnwrap(pixelBuffer)

        let detections = try await detector.detect(pixelBuffer: buffer)
        XCTAssertTrue(detections.allSatisfy { $0.confidence >= LocalEquipmentDetector.defaultConfidenceThreshold })
    }

    func testDistanceFormattingAndSymbolMapping() {
        XCTAssertEqual(EquipmentListView.formattedDistance(meters: 0.42), "42 cm")
        XCTAssertEqual(EquipmentListView.formattedDistance(meters: 1.5), "1.50 m")
        XCTAssertEqual(EquipmentListView.formattedDistance(meters: .nan), "--")

        XCTAssertEqual(EquipmentListView.symbolName(for: "phoropter"), "eyeglasses")
        XCTAssertEqual(EquipmentListView.symbolName(for: "slit lamp"), "eye")
        XCTAssertEqual(EquipmentListView.symbolName(for: "monitor"), "desktopcomputer")
        XCTAssertEqual(EquipmentListView.symbolName(for: "lens drawer"), "cabinet")
    }

    @MainActor
    func testClusteringLookupClearAllAndTimestampMonotonicity() throws {
        let manager = EquipmentClusteringManager()
        let newer = Date(timeIntervalSince1970: 200)
        let older = Date(timeIntervalSince1970: 150)

        let item = manager.registerDetection(
            label: "trial lens set",
            worldPosition: simd_float3(0, 0, -1),
            timestamp: newer
        )
        let id = try XCTUnwrap(item?.id)
        XCTAssertEqual(manager.equipment(withID: id)?.lastSeen, newer)

        // Merging with an older timestamp must preserve the newer lastSeen timestamp.
        let updated = manager.registerDetection(
            label: "trial lens set",
            worldPosition: simd_float3(0.05, 0, -1),
            timestamp: older
        )
        XCTAssertEqual(updated?.lastSeen, newer)
        XCTAssertEqual(updated?.observationCount, 2)

        manager.clearAll()
        XCTAssertTrue(manager.trackedEquipment.isEmpty)
        XCTAssertNil(manager.equipment(withID: id))
    }

    func testMirroredOrientationsAndDepthBoundaries() {
        let box = CGRect(x: 0.1, y: 0.7, width: 0.2, height: 0.2) // center (0.2, 0.8)

        let upMirrored = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .upMirrored)
        XCTAssertEqual(upMirrored?.x ?? -1, 0.8, accuracy: 1e-6)
        XCTAssertEqual(upMirrored?.y ?? -1, 0.2, accuracy: 1e-6)

        let downMirrored = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .downMirrored)
        XCTAssertEqual(downMirrored?.x ?? -1, 0.2, accuracy: 1e-6)
        XCTAssertEqual(downMirrored?.y ?? -1, 0.8, accuracy: 1e-6)

        let rightMirrored = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .rightMirrored)
        XCTAssertEqual(rightMirrored?.x ?? -1, 0.2, accuracy: 1e-6)
        XCTAssertEqual(rightMirrored?.y ?? -1, 0.2, accuracy: 1e-6)

        let leftMirrored = SpatialProjector.normalizedImagePoint(fromVisionBox: box, orientation: .leftMirrored)
        XCTAssertEqual(leftMirrored?.x ?? -1, 0.8, accuracy: 1e-6)
        XCTAssertEqual(leftMirrored?.y ?? -1, 0.8, accuracy: 1e-6)

        let minBoundary = SpatialProjector.selectNearestPosition(
            from: [simd_float3(0, 0, -0.15)],
            cameraPosition: .zero,
            minDepth: 0.15,
            maxDepth: 15.0
        )
        XCTAssertNotNil(minBoundary)

        let maxBoundary = SpatialProjector.selectNearestPosition(
            from: [simd_float3(0, 0, -15.0)],
            cameraPosition: .zero,
            minDepth: 0.15,
            maxDepth: 15.0
        )
        XCTAssertNotNil(maxBoundary)
    }

    @MainActor
    func testAnimatedFocusCancellationAndMeshStyles() {
        XCTAssertEqual(MeshRenderStyle.allCases, [.wireframe, .occlusion])

        let controller = CameraFocusController()
        let item = TrackedEquipment3D(label: "slit lamp", worldPosition: simd_float3(0, 0, -1.5))
        controller.focusOnEquipment(item, duration: 0.75)
        XCTAssertTrue(controller.isVirtualCameraActive)
        XCTAssertEqual(controller.focusedItemID, item.id)

        controller.exitFocusMode()
        XCTAssertFalse(controller.isVirtualCameraActive)
        XCTAssertNil(controller.focusedItemID)
    }

    @MainActor
    func testARSessionConfigurationAndNearestClusterDisambiguation() throws {
        let sessionManager = ARSessionManager()
        let config = sessionManager.makeWorldTrackingConfiguration()
        XCTAssertEqual(config.sceneReconstruction, .meshWithClassification)
        XCTAssertTrue(config.planeDetection.contains(.horizontal))
        XCTAssertTrue(config.planeDetection.contains(.vertical))

        let clustering = EquipmentClusteringManager()
        let leftMonitor = try XCTUnwrap(
            clustering.registerDetection(label: "monitor", worldPosition: simd_float3(0.0, 1.0, -2.0))
        )
        let rightMonitor = try XCTUnwrap(
            clustering.registerDetection(label: "monitor", worldPosition: simd_float3(0.45, 1.0, -2.0))
        )
        XCTAssertEqual(clustering.trackedEquipment.count, 2)

        // Position at x = 0.25 is 0.25 m from leftMonitor and 0.20 m from rightMonitor:
        // both are < 0.35 m, so it must merge with the closer rightMonitor cluster.
        let merged = try XCTUnwrap(
            clustering.registerDetection(label: "monitor", worldPosition: simd_float3(0.25, 1.0, -2.0))
        )
        XCTAssertEqual(clustering.trackedEquipment.count, 2)
        XCTAssertEqual(merged.id, rightMonitor.id)
        XCTAssertEqual(clustering.equipment(withID: leftMonitor.id)?.observationCount, 1)
        XCTAssertEqual(clustering.equipment(withID: rightMonitor.id)?.observationCount, 2)
    }
}
