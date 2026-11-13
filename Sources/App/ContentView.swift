import SwiftUI
import UIKit
import VisualMapperCore
import simd

private enum MapperStage: Equatable {
    case home
    case scanning
    case baking(progress: Double, status: String)
    case viewing(BakedRoomScan)
}

/// Primary root interface supporting the Aunty.cloud clinical dashboard, two-stage
/// live AR capture -> baked 3D Dollhouse inspection, and tap-to-inspect asset knowledge cards.
public struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase

    @StateObject private var sessionManager = ARSessionManager()
    @StateObject private var clusteringManager = EquipmentClusteringManager()
    @StateObject private var focusController = CameraFocusController()

    private let modelBaker = RoomModelBaker()
    private let storageManager = ScanStorageManager()
    private let knowledgeService = HybridEquipmentKnowledgeService()

    @State private var stage: MapperStage = .home
    @State private var environmentMode: EnvironmentMode = .clinicalLab
    @State private var meshRenderStyle: MeshRenderStyle = .wireframe
    @State private var dollhouseRenderMode: DollhouseRenderMode = .photorealistic
    @State private var showCeiling: Bool = false
    @State private var modelLoadErrorMessage: String?
    @State private var scanActionErrorMessage: String?
    @State private var hasConfiguredPipeline = false
    @State private var bakingTask: Task<Void, Never>?

    @State private var inspectedEquipment: TrackedEquipment3D?
    @State private var showSavedScansSheet = false
    @State private var savedScans: [SavedRoomScanMetadata] = []
    @State private var isLoadingSavedScans = false
    @State private var savedScansErrorMessage: String?

    public init() {}

    public var body: some View {
        ZStack {
            if stage == .home {
                ZStack(alignment: .bottom) {
                    HomeView(
                        environmentMode: $environmentMode,
                        savedScans: savedScans,
                        isLoadingScans: isLoadingSavedScans,
                        onStartNewScan: {
                            startNewEnvironmentScan()
                        },
                        onOpenScan: { metadata in
                            openSavedScan(id: metadata.id)
                        },
                        onDeleteScan: { metadata in
                            deleteSavedScan(id: metadata.id)
                        },
                        onRefreshScans: {
                            Task { await refreshSavedScans() }
                        }
                    )

                    if let scanActionErrorMessage {
                        dismissibleErrorBanner(message: scanActionErrorMessage) {
                            self.scanActionErrorMessage = nil
                        }
                        .padding(.horizontal, 16)
                        .padding(.bottom, 16)
                    }
                }
            } else {
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
                            scanCoverageHUD
                                .padding(.horizontal, 16)
                                .padding(.bottom, 8)
                                .zIndex(10)

                            finishScanButton
                                .padding(.horizontal, 16)
                                .padding(.bottom, 10)
                                .zIndex(10)
                        }

                        EquipmentListView(
                            clusteringManager: clusteringManager,
                            focusController: focusController,
                            userPosition: activeViewerPosition,
                            environmentMode: environmentMode,
                            isLiveCaptureMode: stage == .scanning,
                            onInspectEquipment: { item in
                                inspectedEquipment = item
                            }
                        )
                        .zIndex(10)
                    }
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
        .sheet(item: $inspectedEquipment) { item in
            EquipmentDetailSheet(
                item: item,
                environmentMode: environmentMode,
                knowledgeService: knowledgeService
            )
        }
        .task {
            configurePipelineIfNeeded()
            await refreshSavedScans()
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
        case .home:
            EmptyView()
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
                renderMode: dollhouseRenderMode,
                onSelectEquipment: { item in
                    focusController.focusOnEquipment(item)
                    inspectedEquipment = item
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
        case .home, .baking:
            EmptyView()
        case .scanning:
            topScanningBar
        case let .viewing(scan):
            topDollhouseBar(scan: scan)
        }
    }

    private var topScanningBar: some View {
        HStack(spacing: 8) {
            Button {
                returnToHome()
            } label: {
                Image(systemName: "house.fill")
                    .font(.body.weight(.semibold))
                    .padding(8)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Aunty.cloud Home")

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
            .frame(maxWidth: 135)

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
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Button {
                    returnToHome()
                } label: {
                    Image(systemName: "house.fill")
                        .font(.body.weight(.semibold))
                        .padding(8)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Aunty.cloud Home")

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
                    startNewEnvironmentScan()
                } label: {
                    Label("Rescan", systemImage: "camera.viewfinder")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color.accentColor.opacity(0.85), in: Capsule())
                }
                .buttonStyle(.plain)
            }

            HStack(spacing: 8) {
                if scan.textureAtlasData != nil {
                    Picker("Render Mode", selection: $dollhouseRenderMode) {
                        ForEach(DollhouseRenderMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 205)
                } else {
                    Label("Mesh Build Mode", systemImage: "cube.transparent")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }

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
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
    }

    private var scanCoverageHUD: some View {
        let coverage = sessionManager.scanCoverage
        let sectorCount = max(1, coverage.sectorMeshCounts.count)
        let pct = Int((coverage.overallCoverageFraction * 100.0).rounded())

        return HStack(spacing: 12) {
            ZStack {
                Canvas { context, size in
                    let center = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
                    let radius = min(size.width, size.height) * 0.42
                    let stepAngle = (2.0 * Double.pi) / Double(sectorCount)
                    let gap = 0.09

                    for idx in 0..<sectorCount {
                        let start = -Double.pi * 0.5 + Double(idx) * stepAngle + gap
                        let end = -Double.pi * 0.5 + Double(idx + 1) * stepAngle - gap

                        var path = Path()
                        path.addArc(
                            center: center,
                            radius: radius,
                            startAngle: .radians(start),
                            endAngle: .radians(end),
                            clockwise: false
                        )

                        let hasMesh = idx < coverage.sectorMeshCounts.count && coverage.sectorMeshCounts[idx] > 0
                        let hasPhoto = idx < coverage.sectorPhotoCaptured.count && coverage.sectorPhotoCaptured[idx]
                        let isActive = (idx == coverage.activeHeadingSector)

                        let color: Color
                        if hasMesh && hasPhoto {
                            color = .green
                        } else if hasMesh {
                            color = .cyan
                        } else if isActive {
                            color = .orange
                        } else {
                            color = .white.opacity(0.22)
                        }

                        context.stroke(
                            path,
                            with: .color(color),
                            style: StrokeStyle(
                                lineWidth: isActive ? 5.5 : 3.5,
                                lineCap: .round
                            )
                        )
                    }
                }
                .frame(width: 44, height: 44)

                Text("\(pct)%")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text("360° Scan Guide")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.white)

                    Text("\(coverage.capturedKeyframeCount)/\(coverage.maxKeyframeCount) Photos")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(.cyan.opacity(0.22), in: Capsule())
                        .foregroundStyle(.cyan)
                }

                Text(coverage.guidanceHint)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
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

    private func startNewEnvironmentScan() {
        configurePipelineIfNeeded()
        focusController.disableDollhouseOrbit()
        inspectedEquipment = nil
        scanActionErrorMessage = nil
        stage = .scanning
        Task {
            await sessionManager.resetSession()
        }
    }

    private func returnToHome() {
        bakingTask?.cancel()
        bakingTask = nil
        sessionManager.pauseSession()
        focusController.disableDollhouseOrbit()
        inspectedEquipment = nil
        scanActionErrorMessage = nil
        stage = .home
        Task {
            await refreshSavedScans()
        }
    }

    private func finishAndProcessScan() {
        scanActionErrorMessage = nil
        sessionManager.pauseSession()
        focusController.exitFocusMode()

        let snapshots = sessionManager.extractMeshSnapshots()
        let keyframes = sessionManager.extractTextureKeyframes()
        let activeMode = environmentMode

        stage = .baking(progress: 0.05, status: "Consolidating 3D observations (NMS)…")
        let consolidatedEquipment = clusteringManager.consolidateObservations()

        bakingTask?.cancel()
        bakingTask = Task {
            do {
                let bakeResult = try await modelBaker.bakeWithTextures(
                    snapshots: snapshots,
                    keyframes: keyframes,
                    fillPlanarGaps: true
                ) { fraction, status in
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
                    environmentMode: activeMode,
                    geometry: bakeResult.geometry,
                    equipment: consolidatedEquipment,
                    textureAtlasJPEGData: bakeResult.textureAtlasJPEGData
                )
                let bakedScan = BakedRoomScan(
                    metadata: metadata,
                    geometry: bakeResult.geometry,
                    textureAtlasData: bakeResult.textureAtlasJPEGData
                )

                await refreshSavedScans()
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
        inspectedEquipment = nil
        showCeiling = false
        dollhouseRenderMode = (scan.textureAtlasData != nil) ? .photorealistic : .semanticMesh
        scanActionErrorMessage = nil
        focusController.configureDollhouseOrbit(
            boundsMin: scan.geometry.boundsMin,
            boundsMax: scan.geometry.boundsMax,
            duration: 0.0
        )
        stage = .viewing(scan)
    }

    private func refreshSavedScans() async {
        await MainActor.run {
            self.isLoadingSavedScans = true
            self.savedScansErrorMessage = nil
        }
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

    private func presentSavedScans() {
        showSavedScansSheet = true
        Task {
            await refreshSavedScans()
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
                    self.stage = .home
                    self.scanActionErrorMessage = error.localizedDescription
                }
            }
        }
    }

    private func deleteSavedScan(id: UUID) {
        Task {
            do {
                try await storageManager.deleteScan(id: id)
                await refreshSavedScans()
            } catch {
                await MainActor.run {
                    self.savedScansErrorMessage = error.localizedDescription
                }
            }
        }
    }
}
