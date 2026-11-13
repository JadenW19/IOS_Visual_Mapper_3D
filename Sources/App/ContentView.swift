import SwiftUI
import UIKit
import VisualMapperCore
import simd

public struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase

    @StateObject private var sessionManager = ARSessionManager()
    @StateObject private var clusteringManager = EquipmentClusteringManager()
    @StateObject private var focusController = CameraFocusController()

    @State private var meshRenderStyle: MeshRenderStyle = .wireframe
    @State private var modelLoadErrorMessage: String?
    @State private var hasConfiguredPipeline = false

    public init() {}

    public var body: some View {
        ZStack {
            RealitySceneView(
                sessionManager: sessionManager,
                clusteringManager: clusteringManager,
                focusController: focusController,
                meshRenderStyle: meshRenderStyle
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                topStatusBar
                    .zIndex(10)

                Spacer()

                if case let .failed(error) = sessionManager.sessionState {
                    errorOverlay(for: error)
                        .padding(.horizontal, 20)
                    Spacer()
                } else if let modelLoadErrorMessage {
                    modelErrorBanner(message: modelLoadErrorMessage)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }

                EquipmentListView(
                    clusteringManager: clusteringManager,
                    focusController: focusController,
                    userPosition: sessionManager.currentCameraPosition
                )
                .zIndex(10)
            }
        }
        .preferredColorScheme(.dark)
        .task {
            configurePipelineIfNeeded()
            await sessionManager.startSession()
        }
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .active:
                if case .paused = sessionManager.sessionState {
                    Task { await sessionManager.startSession(resetTracking: false) }
                }
            case .background:
                sessionManager.pauseSession()
            case .inactive:
                break
            @unknown default:
                break
            }
        }
    }

    private var topStatusBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 9, height: 9)

                Text(sessionManager.trackingStatusMessage)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())

            Spacer()

            Picker("Mesh Style", selection: $meshRenderStyle) {
                ForEach(MeshRenderStyle.allCases, id: \.self) { style in
                    Text(style.rawValue).tag(style)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 170)

            Button {
                focusController.exitFocusMode()
                Task { await sessionManager.resetSession() }
            } label: {
                Image(systemName: "arrow.counterclockwise.circle.fill")
                    .font(.title3)
                    .padding(6)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Reset scan")
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
    }

    private var statusColor: Color {
        switch sessionManager.sessionState {
        case .running:
            return sessionManager.isInferenceInFlight ? .yellow : .green
        case .requestingPermissions, .idle:
            return .orange
        case .paused, .interrupted:
            return .gray
        case .failed:
            return .red
        }
    }

    private func errorOverlay(for error: VisualMapperError) -> some View {
        VStack(spacing: 14) {
            Image(systemName: error == .cameraPermissionDenied ? "camera.fill" : "sensor.tag.radiowaves.forward.fill")
                .font(.largeTitle)
                .foregroundStyle(.yellow)

            Text(errorTitle(for: error))
                .font(.headline)

            Text(error.localizedDescription)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            if error == .cameraPermissionDenied,
               let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                Link("Open Settings", destination: settingsURL)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private func errorTitle(for error: VisualMapperError) -> String {
        switch error {
        case .cameraPermissionDenied:
            return "Camera Permission Required"
        case .lidarUnsupported:
            return "LiDAR Scanner Unavailable"
        case .modelNotFound, .modelLoadFailed:
            return "Equipment Model Unavailable"
        case .visionInferenceFailed, .arSessionFailed:
            return "AR Session Error"
        }
    }

    private func modelErrorBanner(message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text(message)
                .font(.caption)
                .lineLimit(2)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func configurePipelineIfNeeded() {
        guard !hasConfiguredPipeline else { return }
        hasConfiguredPipeline = true

        focusController.setLiveCameraTransformProvider { [weak sessionManager] in
            sessionManager?.currentCameraTransform ?? matrix_identity_float4x4
        }

        do {
            let detector = try LocalEquipmentDetector()
            sessionManager.configurePipeline(
                detector: detector,
                clusteringManager: clusteringManager
            )
        } catch {
            modelLoadErrorMessage = error.localizedDescription
            sessionManager.configurePipeline(
                detector: nil,
                clusteringManager: clusteringManager
            )
        }
    }
}
