import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import Vision

/// Abstraction for on-device 2D equipment detection services.
public protocol EquipmentDetecting: Sendable {
    func detect(pixelBuffer: CVPixelBuffer) async throws -> [Detection2D]
}

/// Actor-isolated CoreML + Vision inference service backed by `yolov8s-worldv2.mlpackage`.
public actor LocalEquipmentDetector: EquipmentDetecting {
    /// Value-type representation of a single classification candidate inside an object observation.
    public struct RawClassification: Sendable, Equatable {
        public let identifier: String
        public let confidence: Float

        public init(identifier: String, confidence: Float) {
            self.identifier = identifier
            self.confidence = confidence
        }
    }

    /// Value-type representation of a raw Vision object observation for pure deterministic testing.
    public struct RawObjectObservation: Sendable, Equatable {
        public let boundingBox: CGRect
        public let labels: [RawClassification]

        public init(boundingBox: CGRect, labels: [RawClassification]) {
            self.boundingBox = boundingBox
            self.labels = labels
        }
    }

    /// Default minimum confidence required to retain a detection.
    public static let defaultConfidenceThreshold: Float = 0.40

    private struct UncheckedPixelBufferBox: @unchecked Sendable {
        let buffer: CVPixelBuffer
    }

    private let visionModel: VNCoreMLModel
    public let confidenceThreshold: Float

    /// Initializes the detector from an explicit URL or by resolving `yolov8s-worldv2` / `yolov8s_worldv2`
    /// in the supplied bundle. Execution compute units are set to `.all` to prioritize the Apple Neural Engine.
    public init(
        modelURL: URL? = nil,
        bundle: Bundle = .main,
        confidenceThreshold: Float = LocalEquipmentDetector.defaultConfidenceThreshold
    ) throws {
        self.confidenceThreshold = confidenceThreshold.isFinite
            ? max(0.0, min(1.0, confidenceThreshold))
            : Self.defaultConfidenceThreshold

        let resolvedURL = try Self.resolveModelURL(explicitURL: modelURL, bundle: bundle)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        do {
            let mlModel = try MLModel(contentsOf: resolvedURL, configuration: configuration)
            self.visionModel = try VNCoreMLModel(for: mlModel)
        } catch {
            throw VisualMapperError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Runs object detection on the given `CVPixelBuffer` assuming `.up` buffer orientation.
    nonisolated public func detect(pixelBuffer: CVPixelBuffer) async throws -> [Detection2D] {
        try await detect(pixelBuffer: pixelBuffer, orientation: .up)
    }

    /// Runs object detection on the given `CVPixelBuffer` using the specified image orientation
    /// so Vision evaluates the frame upright relative to the physical camera orientation.
    nonisolated public func detect(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) async throws -> [Detection2D] {
        let box = UncheckedPixelBufferBox(buffer: pixelBuffer)
        return try await performDetection(on: box, orientation: orientation)
    }

    private func performDetection(
        on box: UncheckedPixelBufferBox,
        orientation: CGImagePropertyOrientation
    ) throws -> [Detection2D] {
        try Task.checkCancellation()

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(
            cvPixelBuffer: box.buffer,
            orientation: orientation,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            throw VisualMapperError.visionInferenceFailed(error.localizedDescription)
        }

        let observations = request.results as? [VNRecognizedObjectObservation] ?? []
        return Self.parseObservations(observations, confidenceThreshold: confidenceThreshold)
    }

    /// Parses Vision `VNRecognizedObjectObservation` results into `Detection2D` values,
    /// discarding detections below `confidenceThreshold` (default 0.40).
    public static func parseObservations(
        _ observations: [VNRecognizedObjectObservation],
        confidenceThreshold: Float = defaultConfidenceThreshold
    ) -> [Detection2D] {
        let rawObservations = observations.map { observation in
            RawObjectObservation(
                boundingBox: observation.boundingBox,
                labels: observation.labels.map {
                    RawClassification(identifier: $0.identifier, confidence: $0.confidence)
                }
            )
        }
        return parseObservations(raw: rawObservations, confidenceThreshold: confidenceThreshold)
    }

    /// Parses raw object observations into `Detection2D` values, selecting the highest-confidence
    /// finite classification label and discarding detections below `confidenceThreshold` (default 0.40).
    public static func parseObservations(
        raw observations: [RawObjectObservation],
        confidenceThreshold: Float = defaultConfidenceThreshold
    ) -> [Detection2D] {
        let threshold = confidenceThreshold.isFinite ? confidenceThreshold : defaultConfidenceThreshold
        return observations.compactMap { observation in
            let validLabels = observation.labels.filter { $0.confidence.isFinite }
            guard let topClassification = validLabels.max(by: { $0.confidence < $1.confidence }) else {
                return nil
            }

            let trimmedLabel = topClassification.identifier.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedLabel.isEmpty else {
                return nil
            }

            let score = topClassification.confidence
            guard score >= threshold else {
                return nil
            }

            let clampedBox = clampNormalizedRect(observation.boundingBox)
            guard clampedBox.width > 0, clampedBox.height > 0 else {
                return nil
            }

            return Detection2D(
                boundingBox: clampedBox,
                label: trimmedLabel,
                confidence: score
            )
        }
    }

    /// Pure utility for clamping normalized bounding boxes to `[0, 1] x [0, 1]`.
    public static func clampNormalizedRect(_ rect: CGRect) -> CGRect {
        guard rect.origin.x.isFinite,
              rect.origin.y.isFinite,
              rect.size.width.isFinite,
              rect.size.height.isFinite else {
            return .zero
        }

        let minX = max(0.0, min(1.0, rect.minX))
        let minY = max(0.0, min(1.0, rect.minY))
        let maxX = max(minX, min(1.0, rect.maxX))
        let maxY = max(minY, min(1.0, rect.maxY))
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private static func resolveModelURL(explicitURL: URL?, bundle: Bundle) throws -> URL {
        if let explicitURL {
            if explicitURL.pathExtension == "mlpackage" {
                return try MLModel.compileModel(at: explicitURL)
            }
            return explicitURL
        }

        let candidateNames = ["yolov8s-worldv2", "yolov8s_worldv2"]
        for name in candidateNames {
            if let compiledURL = bundle.url(forResource: name, withExtension: "mlmodelc") {
                return compiledURL
            }
        }

        for name in candidateNames {
            if let packageURL = bundle.url(forResource: name, withExtension: "mlpackage") {
                return try MLModel.compileModel(at: packageURL)
            }
        }

        throw VisualMapperError.modelNotFound("yolov8s-worldv2")
    }
}
