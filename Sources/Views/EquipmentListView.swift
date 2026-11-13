import SwiftUI
import UIKit
import simd

/// Responsive bottom drawer displaying the live roster of clustered 3D clinic equipment.
public struct EquipmentListView: View {
    @ObservedObject var clusteringManager: EquipmentClusteringManager
    @ObservedObject var focusController: CameraFocusController
    let userPosition: simd_float3

    @State private var isExpanded: Bool = true
    @GestureState private var dragOffset: CGFloat = 0

    public init(
        clusteringManager: EquipmentClusteringManager,
        focusController: CameraFocusController,
        userPosition: simd_float3
    ) {
        self.clusteringManager = clusteringManager
        self.focusController = focusController
        self.userPosition = userPosition
    }

    private var sortedEquipment: [TrackedEquipment3D] {
        clusteringManager.trackedEquipment.sorted { lhs, rhs in
            let distA = simd_distance(userPosition, lhs.worldPosition)
            let distB = simd_distance(userPosition, rhs.worldPosition)
            if abs(distA - distB) > 0.05 {
                return distA < distB
            }
            return lhs.lastSeen > rhs.lastSeen
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            drawerHeader
            if isExpanded {
                Divider()
                    .overlay(Color.white.opacity(0.15))
                equipmentContent
            }
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
        )
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .offset(y: max(0, dragOffset))
        .animation(.spring(response: 0.32, dampingFraction: 0.86), value: isExpanded)
        .animation(.easeInOut(duration: 0.2), value: focusController.focusedItemID)
    }

    private var drawerHeader: some View {
        VStack(spacing: 8) {
            Capsule()
                .fill(Color.secondary.opacity(0.55))
                .frame(width: 38, height: 5)
                .padding(.top, 10)

            HStack(spacing: 10) {
                Label("Detected Equipment", systemImage: "viewfinder")
                    .font(.headline)
                    .lineLimit(1)

                Text("\(clusteringManager.trackedEquipment.count)")
                    .font(.caption.weight(.bold))
                    .monospacedDigit()
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.accentColor.opacity(0.22), in: Capsule())

                Spacer()

                if focusController.isVirtualCameraActive {
                    Button {
                        triggerSelectionHaptic()
                        focusController.exitFocusMode()
                    } label: {
                        Label("Live AR", systemImage: "arkit")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color.yellow.opacity(0.24), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }

                Button {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.86)) {
                        isExpanded.toggle()
                    }
                } label: {
                    Image(systemName: isExpanded ? "chevron.down.circle.fill" : "chevron.up.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(isExpanded ? "Collapse equipment drawer" : "Expand equipment drawer")
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 10)
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 12)
                .updating($dragOffset) { value, state, _ in
                    if value.translation.height > 0 {
                        state = value.translation.height * 0.45
                    }
                }
                .onEnded { value in
                    if value.translation.height > 40 {
                        withAnimation(.spring(response: 0.32, dampingFraction: 0.86)) {
                            isExpanded = false
                        }
                    } else if value.translation.height < -30 {
                        withAnimation(.spring(response: 0.32, dampingFraction: 0.86)) {
                            isExpanded = true
                        }
                    }
                }
        )
        .onTapGesture {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.86)) {
                isExpanded.toggle()
            }
        }
    }

    @ViewBuilder
    private var equipmentContent: some View {
        if sortedEquipment.isEmpty {
            VStack(spacing: 6) {
                Image(systemName: "camera.metering.center.weighted")
                    .font(.title2)
                    .foregroundStyle(.secondary)
                Text("Scanning clinic space…")
                    .font(.subheadline.weight(.medium))
                Text("Point the iPhone LiDAR camera at equipment (e.g. phoropter, slit lamp, autorefractor).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .padding(.horizontal, 16)
        } else {
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(sortedEquipment) { item in
                        let distance = simd_distance(userPosition, item.worldPosition)
                        let isFocused = (focusController.focusedItemID == item.id)

                        EquipmentRowView(
                            item: item,
                            distanceMeters: distance,
                            isFocused: isFocused
                        ) {
                            triggerSelectionHaptic()
                            focusController.focusOnEquipment(item)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            }
            .frame(maxHeight: 230)
        }
    }

    private func triggerSelectionHaptic() {
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        generator.prepare()
    }

    /// Formats a 3D Euclidean distance for display in clinical room scanning rows.
    nonisolated public static func formattedDistance(meters: Float) -> String {
        guard meters.isFinite, meters >= 0 else { return "--" }
        if meters < 1.0 {
            let centimeters = Int((meters * 100.0).rounded())
            return "\(centimeters) cm"
        }
        return String(format: "%.2f m", meters)
    }

    /// Maps detected medical/lab equipment labels to SF Symbols.
    nonisolated public static func symbolName(for label: String) -> String {
        let normalized = label.lowercased()
        if normalized.contains("drawer") || normalized.contains("cabinet") {
            return "cabinet"
        } else if normalized.contains("phoropter") || normalized.contains("lens") {
            return "eyeglasses"
        } else if normalized.contains("slit") || normalized.contains("autorefractor") {
            return "eye"
        } else if normalized.contains("monitor") || normalized.contains("desktop") || normalized.contains("computer") {
            return "desktopcomputer"
        } else if normalized.contains("chair") {
            return "chair.lounge"
        }
        return "cube.transparent"
    }
}

private struct EquipmentRowView: View {
    let item: TrackedEquipment3D
    let distanceMeters: Float
    let isFocused: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                Image(systemName: EquipmentListView.symbolName(for: item.label))
                    .font(.title3)
                    .foregroundStyle(isFocused ? Color.yellow : Color.cyan)
                    .frame(width: 36, height: 36)
                    .background(
                        (isFocused ? Color.yellow : Color.cyan).opacity(0.16),
                        in: RoundedRectangle(cornerRadius: 10, style: .continuous)
                    )

                VStack(alignment: .leading, spacing: 2) {
                    Text(item.label.capitalized)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    HStack(spacing: 8) {
                        Label("\(item.observationCount) obs", systemImage: "eye.fill")
                            .font(.caption2)
                            .foregroundStyle(.secondary)

                        Text("•")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)

                        Text(EquipmentListView.formattedDistance(meters: distanceMeters))
                            .font(.caption.weight(.medium))
                            .monospacedDigit()
                            .foregroundStyle(isFocused ? Color.yellow : Color.secondary)
                    }
                }

                Spacer()

                Image(systemName: isFocused ? "scope" : "location.viewfinder")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(isFocused ? Color.yellow : Color.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(isFocused ? Color.yellow.opacity(0.14) : Color.primary.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(
                        isFocused ? Color.yellow.opacity(0.65) : Color.clear,
                        lineWidth: 1.2
                    )
            )
        }
        .buttonStyle(.plain)
    }
}
