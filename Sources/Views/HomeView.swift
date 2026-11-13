import SwiftUI

/// Clean clinical-grade Aunty.cloud dashboard serving as the application's entry hub,
/// separating offline spatial scan library management from live AR sensor capture.
public struct HomeView: View {
    @Binding public var environmentMode: EnvironmentMode
    public let savedScans: [SavedRoomScanMetadata]
    public let isLoadingScans: Bool
    public let isLiDARReady: Bool
    public let onStartNewScan: () -> Void
    public let onOpenScan: (SavedRoomScanMetadata) -> Void
    public let onDeleteScan: (SavedRoomScanMetadata) -> Void
    public let onRefreshScans: () -> Void

    private let gridColumns = [
        GridItem(.adaptive(minimum: 158), spacing: 12)
    ]

    public init(
        environmentMode: Binding<EnvironmentMode>,
        savedScans: [SavedRoomScanMetadata],
        isLoadingScans: Bool,
        isLiDARReady: Bool = ARSessionManager.isLiDARSupported,
        onStartNewScan: @escaping () -> Void,
        onOpenScan: @escaping (SavedRoomScanMetadata) -> Void,
        onDeleteScan: @escaping (SavedRoomScanMetadata) -> Void,
        onRefreshScans: @escaping () -> Void
    ) {
        self._environmentMode = environmentMode
        self.savedScans = savedScans
        self.isLoadingScans = isLoadingScans
        self.isLiDARReady = isLiDARReady
        self.onStartNewScan = onStartNewScan
        self.onOpenScan = onOpenScan
        self.onDeleteScan = onDeleteScan
        self.onRefreshScans = onRefreshScans
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    brandHeaderCard
                    environmentProfileCard
                    primaryScanCTA
                    savedModelsSection
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 28)
            }
            .background(
                LinearGradient(
                    colors: [
                        Color(red: 0.05, green: 0.07, blue: 0.11),
                        Color(red: 0.03, green: 0.04, blue: 0.07)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()
            )
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var brandHeaderCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color.cyan.opacity(0.85), Color.blue.opacity(0.8)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 44, height: 44)
                    Image(systemName: "cube.transparent.fill")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Aunty.cloud")
                            .font(.title2.weight(.bold))
                            .foregroundStyle(.white)

                        Text("OFFLINE")
                            .font(.caption2.weight(.bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.cyan.opacity(0.22), in: Capsule())
                            .foregroundStyle(.cyan)
                    }
                    Text("Spatial Environment & Clinical Asset Mapper")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            HStack(spacing: 8) {
                Circle()
                    .fill(isLiDARReady ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)

                Text(isLiDARReady ? "LiDAR & Neural Engine Ready" : "Neural Engine Ready • LiDAR Device Required")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(isLiDARReady ? Color.green : Color.orange)

                Spacer()

                Label("On-Device", systemImage: "lock.shield.fill")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
        )
    }

    private var environmentProfileCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Inspection Profile", systemImage: "slider.horizontal.3")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(environmentMode.shortTitle)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.cyan)
            }

            Picker("Environment Mode", selection: $environmentMode) {
                ForEach(EnvironmentMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(environmentMode.subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
        )
    }

    private var primaryScanCTA: some View {
        Button {
            onStartNewScan()
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "camera.viewfinder")
                    .font(.title3.weight(.semibold))

                VStack(alignment: .leading, spacing: 2) {
                    Text("Start New Environment Scan")
                        .font(.headline.weight(.semibold))
                    Text("Capture lightweight 3D frames, then bake single-identity Dollhouse pins")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.85))
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.subheadline.weight(.bold))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 16)
            .frame(maxWidth: .infinity)
            .background(
                LinearGradient(
                    colors: [Color(red: 0.04, green: 0.54, blue: 0.78), Color(red: 0.08, green: 0.36, blue: 0.72)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 18, style: .continuous)
            )
            .foregroundStyle(.white)
            .shadow(color: Color.cyan.opacity(0.22), radius: 12, y: 4)
        }
        .buttonStyle(.plain)
    }

    private var savedModelsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Saved Spatial Models", systemImage: "folder.fill")
                    .font(.headline)

                Text("\(savedScans.count)")
                    .font(.caption.weight(.bold))
                    .monospacedDigit()
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.cyan.opacity(0.20), in: Capsule())

                Spacer()

                Button {
                    onRefreshScans()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Refresh saved models")
            }

            if isLoadingScans {
                ProgressView("Loading saved models…")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 32)
            } else if savedScans.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "cube.transparent")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No Baked Models Yet")
                        .font(.subheadline.weight(.semibold))
                    Text("Tap Start New Environment Scan to reconstruct a 3D Dollhouse space with consolidated equipment pins.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 28)
                .padding(.horizontal, 16)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            } else {
                LazyVGrid(columns: gridColumns, spacing: 12) {
                    ForEach(savedScans) { scan in
                        SavedSpatialModelCard(
                            scan: scan,
                            onOpen: { onOpenScan(scan) },
                            onDelete: { onDeleteScan(scan) }
                        )
                    }
                }
            }
        }
    }
}

private struct SavedSpatialModelCard: View {
    let scan: SavedRoomScanMetadata
    let onOpen: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Button(action: onOpen) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    ZStack {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Color.cyan.opacity(0.18))
                            .frame(width: 38, height: 38)
                        Image(systemName: "cube.fill")
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.cyan)
                    }

                    Spacer()

                    Button(role: .destructive, action: onDelete) {
                        Image(systemName: "trash")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(6)
                            .background(Color.white.opacity(0.08), in: Circle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Delete \(scan.name)")
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(scan.name)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    Text(scan.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Divider()
                    .overlay(Color.white.opacity(0.12))

                HStack(spacing: 6) {
                    Label("\(scan.equipment.count) assets", systemImage: "mappin.and.ellipse")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.cyan)
                    Spacer()
                    Text("\(scan.triangleCount.formatted()) tris")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}
