import SwiftUI
import UIKit
import VisualMapperCore
import simd

private enum MapperStage: Equatable {
    case scanning
    case baking(progress: Double, status: String)
    case viewing(BakedRoomScan)
}

/// Primary root interface supporting two-stage live AR room scanning -> baked 3D Dollhouse
/// inspection, persisted scan browsing, and floating equipment navigation.
public struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase

    @StateObject private var sessionManager = ARSessionManager()
    @StateObject private var clusteringManager = EquipmentClusteringManager()
    @StateObject private var focusController = CameraFocusController()

    private let modelBaker = RoomModelBaker()
    private let storageManager = ScanStorageManager()

    @State private var stage: MapperStage = .scanning
    @State private var meshRenderStyle: MeshRenderStyle = .wireframe
    @State private var showCeiling: Bool = false
    @State private var modelLoadErrorMessage: String?
    @State private var scanActionErrorMessage: String?
    @State private var hasConfiguredPipeline = false
    @State private var bakingTask: Task<Void, Never>?

    @State private var showSavedScansSheet = false
    @State private var savedScans: [SavedRoomScanMetadata] = []
    @State private var isLoadingSavedScans = false
    @State private var savedScansErrorMessage: String?

    public init() {}

    public var body: some View {
        ZStack {
            sceneContent
                .ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                    .zIndex(10)

                Spacer()

                if case let .baking(progress, status) = stage {
                    bakingProgressOverlay(progress: progress, status: status)
                        .padding(.horizontal, 24)
                    Spacer()
                } else {
                    if case let .failed(error) = sessionManager.sessionState,
                       stage == .scanning {
                        errorOverlay(for: error)
                            .padding(.horizontal, 20)
                        Spacer()
                    } else if let scanActionErrorMessage {
                        dismissibleErrorBanner(message: scanActionErrorMessage) {
                            self.scanActionErrorMessage = nil
                        }
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                    } else if let modelLoadErrorMessage, stage == .scanning {
                        modelErrorBanner(message: modelLoadErrorMessage)
                            .padding(.horizontal, 16)
                            .padding(.bottom, 8)
                    }

                    if stage == .scanning {
                        finishScanButton
                            .padding(.horizontal, 16)
                            .padding(.bottom, 10)
                            .zIndex(10)
                    }

                    EquipmentListView(
                        clusteringManager: clusteringManager,
                        focusController: focusController,
                        userPosition: activeViewerPosition
                    )
                    .zIndex(10)
                }
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showSavedScansSheet) {
            SavedScansSheet(
                scans: savedScans,
                isLoading: isLoadingSavedScans,
                errorMessage: savedScansErrorMessage,
                onSelectScan: { metadata in
                    openSavedScan(id: metadata.id)
                },
                onDeleteScan: { metadata in
                    deleteSavedScan(id: metadata.id)
                }
            )
        }
        .task {
            configurePipelineIfNeeded()
            await sessionManager.startSession()
        }
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .active:
                if stage == .scanning, case .paused = sessionManager.sessionState {
                    Task { await sessionManager.startSession(resetTracking: false) }
                }
            case .background:
                if stage == .scanning {
                    sessionManager.pauseSession()
                }
            case .inactive:
                break
            @unknown default:
                break
            }
        }
    }

    @ViewBuilder
    private var sceneContent: some View {
        switch stage {
        case .scanning, .baking:
            RealitySceneView(
                sessionManager: sessionManager,
                clusteringManager: clusteringManager,
                focusController: focusController,
                meshRenderStyle: meshRenderStyle
            )
        case let .viewing(scan):
            DollhouseViewerView(
                scan: scan,
                equipment: clusteringManager.trackedEquipment,
                focusController: focusController,
                showCeiling: showCeiling,
                onSelectEquipment: { item in
                    focusController.focusOnEquipment(item)
                }
            )
        }
    }

    private var activeViewerPosition: simd_float3 {
        if case .viewing = stage {
            return CameraFocusController.extractPosition(
                from: focusController.virtualCameraTransform.matrix
            )
        }
        return sessionManager.currentCameraPosition
    }

    @ViewBuilder
    private var topBar: some View {
        switch stage {
        case .scanning:
            topScanningBar
        case .baking:
            EmptyView()
        case let .viewing(scan):
            topDollhouseBar(scan: scan)
        }
    }

    private var topScanningBar: some View {
        HStack(spacing: 8) {
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

            Spacer(minLength: 4)

            Picker("Mesh Style", selection: $meshRenderStyle) {
                ForEach(MeshRenderStyle.allCases, id: \.self) { style in
                    Text(style.rawValue).tag(style)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 145)

            Button {
                presentSavedScans()
            } label: {
                Image(systemName: "folder.fill")
                    .font(.body.weight(.semibold))
                    .padding(8)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Saved scans")

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

    private func topDollhouseBar(scan: BakedRoomScan) -> some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(scan.metadata.name)
                    .font(.caption.weight(.bold))
                    .lineLimit(1)
                Text("Dollhouse 3D • \(scan.geometry.triangleCount.formatted()) tris")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())

            Spacer(minLength: 4)

            Button {
                showCeiling.toggle()
            } label: {
                Label("Ceiling", systemImage: showCeiling ? "eye.fill" : "eye.slash")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            .buttonStyle(.plain)

            Button {
                presentSavedScans()
            } label: {
                Image(systemName: "folder.fill")
                    .font(.body.weight(.semibold))
                    .padding(8)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Saved scans")

            Button {
                returnToLiveScanner()
            } label: {
                Label("New Scan", systemImage: "camera.viewfinder")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.accentColor.opacity(0.85), in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
    }

    private var finishScanButton: some View {
        Button {
            finishAndProcessScan()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "cube.transparent.fill")
                    .font(.subheadline.weight(.semibold))
                Text("Finish & Process Scan")
                    .font(.subheadline.weight(.semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.28), radius: 8, y: 3)
        }
        .buttonStyle(.plain)
    }

    private func bakingProgressOverlay(progress: Double, status: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "cube.transparent.fill")
                .font(.largeTitle)
                .foregroundStyle(.cyan)

            Text("Processing 3D Dollhouse")
                .font(.headline)

            ProgressView(value: min(max(progress, 0.0), 1.0))
                .progressViewStyle(.linear)

            Text("\(status) (\(Int((progress * 100).rounded()))%)")
                .font(.caption)
                .foregroundStyle(.secondary)

            Button("Cancel") {
                bakingTask?.cancel()
                bakingTask = nil
                stage = .scanning
                Task { await sessionManager.startSession(resetTracking: false) }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(22)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
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
        case .meshBakingFailed:
            return "3D Model Baking Failed"
        case .scanStorageFailed:
            return "Scan Storage Error"
        }
    }

    private func dismissibleErrorBanner(message: String, onDismiss: @escaping () -> Void) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text(message)
                .font(.caption)
                .lineLimit(3)
            Spacer(minLength: 4)
            Button {
                onDismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
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

    // MARK: - Scan & Bake Workflow Actions

    private func finishAndProcessScan() {
        scanActionErrorMessage = nil
        sessionManager.pauseSession()
        focusController.exitFocusMode()

        let snapshots = sessionManager.extractMeshSnapshots()
        let equipmentSnapshot = clusteringManager.trackedEquipment

        stage = .baking(progress: 0.05, status: "Preparing LiDAR mesh buffers…")

        bakingTask?.cancel()
        bakingTask = Task {
            do {
                let geometry = try await modelBaker.bake(snapshots: snapshots) { fraction, status in
                    Task { @MainActor in
                        if case .baking = self.stage {
                            self.stage = .baking(progress: fraction, status: status)
                        }
                    }
                }

                try Task.checkCancellation()

                await MainActor.run {
                    if case .baking = self.stage {
                        self.stage = .baking(progress: 0.96, status: "Saving Dollhouse 3D scan…")
                    }
                }

                let defaultName = "Room Scan • \(Date().formatted(date: .abbreviated, time: .shortened))"
                let metadata = try await storageManager.saveScan(
                    name: defaultName,
                    geometry: geometry,
                    equipment: equipmentSnapshot
                )
                let bakedScan = BakedRoomScan(metadata: metadata, geometry: geometry)

                await MainActor.run {
                    self.enterDollhouseViewer(with: bakedScan)
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.stage = .scanning
                }
            } catch {
                await MainActor.run {
                    self.stage = .scanning
                    self.scanActionErrorMessage = error.localizedDescription
                    Task { await self.sessionManager.startSession(resetTracking: false) }
                }
            }
        }
    }

    @MainActor
    private func enterDollhouseViewer(with scan: BakedRoomScan) {
        clusteringManager.loadEquipment(scan.metadata.equipment)
        showCeiling = false
        scanActionErrorMessage = nil
        focusController.configureDollhouseOrbit(
            boundsMin: scan.geometry.boundsMin,
            boundsMax: scan.geometry.boundsMax,
            duration: 0.0
        )
        stage = .viewing(scan)
    }

    private func returnToLiveScanner() {
        focusController.disableDollhouseOrbit()
        scanActionErrorMessage = nil
        stage = .scanning
        Task {
            await sessionManager.resetSession()
        }
    }

    private func presentSavedScans() {
        savedScansErrorMessage = nil
        isLoadingSavedScans = true
        showSavedScansSheet = true

        Task {
            do {
                let list = try await storageManager.listSavedScans()
                await MainActor.run {
                    self.savedScans = list
                    self.isLoadingSavedScans = false
                }
            } catch {
                await MainActor.run {
                    self.savedScansErrorMessage = error.localizedDescription
                    self.isLoadingSavedScans = false
                }
            }
        }
    }

    private func openSavedScan(id: UUID) {
        sessionManager.pauseSession()
        bakingTask?.cancel()
        bakingTask = nil
        stage = .baking(progress: 0.5, status: "Loading saved 3D model…")

        Task {
            do {
                let loaded = try await storageManager.loadScan(id: id)
                await MainActor.run {
                    self.enterDollhouseViewer(with: loaded)
                }
            } catch {
                await MainActor.run {
                    self.stage = .scanning
                    self.scanActionErrorMessage = error.localizedDescription
                    Task { await self.sessionManager.startSession(resetTracking: false) }
                }
            }
        }
    }

    private func deleteSavedScan(id: UUID) {
        Task {
            do {
                try await storageManager.deleteScan(id: id)
                let updated = try await storageManager.listSavedScans()
                await MainActor.run {
                    self.savedScans = updated
                }
            } catch {
                await MainActor.run {
                    self.savedScansErrorMessage = error.localizedDescription
                }
            }
        }
    }
}
