import SwiftUI

/// Presents locally persisted baked Dollhouse scans with load and swipe-to-delete actions.
public struct SavedScansSheet: View {
    @Environment(\.dismiss) private var dismiss

    public let scans: [SavedRoomScanMetadata]
    public let isLoading: Bool
    public let errorMessage: String?
    public let onSelectScan: (SavedRoomScanMetadata) -> Void
    public let onDeleteScan: (SavedRoomScanMetadata) -> Void

    public init(
        scans: [SavedRoomScanMetadata],
        isLoading: Bool,
        errorMessage: String?,
        onSelectScan: @escaping (SavedRoomScanMetadata) -> Void,
        onDeleteScan: @escaping (SavedRoomScanMetadata) -> Void
    ) {
        self.scans = scans
        self.isLoading = isLoading
        self.errorMessage = errorMessage
        self.onSelectScan = onSelectScan
        self.onDeleteScan = onDeleteScan
    }

    public var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView("Loading saved rooms…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let errorMessage {
                    ContentUnavailableView(
                        "Unable to Load Scans",
                        systemImage: "exclamationmark.triangle",
                        description: Text(errorMessage)
                    )
                } else if scans.isEmpty {
                    ContentUnavailableView(
                        "No Saved 3D Scans",
                        systemImage: "cube.transparent",
                        description: Text("Scan a room and tap Finish & Process Scan to bake and save an interactive Dollhouse 3D model.")
                    )
                } else {
                    List {
                        ForEach(scans) { scan in
                            Button {
                                onSelectScan(scan)
                                dismiss()
                            } label: {
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text(scan.name)
                                            .font(.headline)
                                            .foregroundStyle(.primary)
                                        Spacer()
                                        Text("\(scan.equipment.count) items")
                                            .font(.caption.weight(.semibold))
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 3)
                                            .background(Color.accentColor.opacity(0.16), in: Capsule())
                                    }

                                    HStack(spacing: 12) {
                                        Label(
                                            scan.createdAt.formatted(date: .abbreviated, time: .shortened),
                                            systemImage: "calendar"
                                        )
                                        Label(
                                            "\(scan.triangleCount.formatted()) tris",
                                            systemImage: "skew"
                                        )
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                }
                                .padding(.vertical, 4)
                            }
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    onDeleteScan(scan)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                    #if os(iOS)
                    .listStyle(.insetGrouped)
                    #endif
                }
            }
            .navigationTitle("Saved Rooms")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
    }
}
