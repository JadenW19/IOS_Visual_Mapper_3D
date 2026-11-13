import SwiftUI

/// Interactive asset inspection sheet surfacing clinical or household context,
/// operational & safety guidance, 3D detection vote telemetry, and HL7/FHIR/UDI metadata.
public struct EquipmentDetailSheet: View {
    @Environment(\.dismiss) private var dismiss

    public let item: TrackedEquipment3D
    public let environmentMode: EnvironmentMode
    public let knowledgeService: any EquipmentKnowledgeProviding

    @State private var profile: EquipmentIntelligenceProfile?
    @State private var isLoading = true
    @State private var loadError: String?

    public init(
        item: TrackedEquipment3D,
        environmentMode: EnvironmentMode,
        knowledgeService: any EquipmentKnowledgeProviding = HybridEquipmentKnowledgeService()
    ) {
        self.item = item
        self.environmentMode = environmentMode
        self.knowledgeService = knowledgeService
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if isLoading {
                        ProgressView("Retrieving Aunty.cloud asset intelligence…")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 48)
                    } else if let profile {
                        headerCard(profile: profile)
                        spatialMetricsRow
                        infoSection(
                            title: environmentMode == .clinicalLab ? "Clinical Role & Function" : "Spatial & Functional Summary",
                            icon: "stethoscope",
                            text: profile.clinicalOrFunctionalSummary
                        )
                        infoSection(
                            title: "Operational Guidelines",
                            icon: "checklist",
                            text: profile.operationalGuidelines
                        )
                        infoSection(
                            title: "Maintenance & Safety Notes",
                            icon: "shield.checkered",
                            text: profile.maintenanceOrSafetyNotes
                        )
                        if !item.labelVotes.isEmpty {
                            voteConfidenceCard
                        }
                        enterpriseMetadataCard(profile: profile)
                    } else if let loadError {
                        ContentUnavailableView(
                            "Knowledge Lookup Failed",
                            systemImage: "exclamationmark.triangle",
                            description: Text(loadError)
                        )
                    }
                }
                .padding(16)
            }
            .navigationTitle("Asset Inspector")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .task(id: "\(item.id.uuidString)-\(environmentMode.rawValue)") {
                await loadProfile()
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    private func loadProfile() async {
        isLoading = true
        loadError = nil
        do {
            profile = try await knowledgeService.fetchContext(for: item, mode: environmentMode)
        } catch {
            loadError = error.localizedDescription
        }
        isLoading = false
    }

    private func headerCard(profile: EquipmentIntelligenceProfile) -> some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.cyan.opacity(0.20))
                    .frame(width: 52, height: 52)
                Image(systemName: EquipmentListView.symbolName(for: item.label))
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.cyan)
            }

            VStack(alignment: .leading, spacing: 5) {
                Text(profile.displayName)
                    .font(.headline.weight(.bold))

                Text(profile.category)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack(spacing: 6) {
                    Label(environmentMode.shortTitle, systemImage: "building.2.crop.circle")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.18), in: Capsule())

                    Label(profile.source.rawValue, systemImage: "cpu")
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.green.opacity(0.16), in: Capsule())
                        .foregroundStyle(.green)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var spatialMetricsRow: some View {
        HStack(spacing: 10) {
            metricTile(
                title: "Confidence",
                value: "\(Int((item.averageConfidence * 100).rounded()))%",
                icon: "checkmark.seal.fill"
            )
            metricTile(
                title: "Observations",
                value: "\(item.observationCount) hits",
                icon: "camera.metering.multispot"
            )
            metricTile(
                title: "Est. Width",
                value: item.estimatedWidthMeters.map { String(format: "%.2f m", $0) } ?? "--",
                icon: "ruler"
            )
        }
    }

    private func metricTile(title: String, value: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(title, systemImage: icon)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .monospacedDigit()
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func infoSection(title: String, icon: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.cyan)
            Text(text)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var voteConfidenceCard: some View {
        let sortedVotes = item.labelVotes.sorted { $0.value > $1.value }
        return VStack(alignment: .leading, spacing: 8) {
            Label("3D NMS Consensus Votes", systemImage: "chart.bar.xaxis")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            ForEach(sortedVotes, id: \.key) { label, score in
                HStack {
                    Text(label.capitalized)
                        .font(.caption.weight(label.lowercased() == item.label.lowercased() ? .bold : .regular))
                    Spacer()
                    Text(String(format: "%.2f", score))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func enterpriseMetadataCard(profile: EquipmentIntelligenceProfile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("HL7 / FHIR & Asset Integration", systemImage: "barcode.viewfinder")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            metadataRow(key: "Asset Tag", value: profile.assetTagID)
            metadataRow(key: "FHIR Device", value: profile.fhirDeviceCode)
            metadataRow(key: "GS1 UDI", value: profile.udibaBarcode)
            metadataRow(
                key: "3D Centroid",
                value: String(format: "(%.2f, %.2f, %.2f) m", item.worldPosition.x, item.worldPosition.y, item.worldPosition.z)
            )
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func metadataRow(key: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(key)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 82, alignment: .leading)
            Text(value)
                .font(.caption.monospaced())
                .foregroundStyle(.primary)
                .textSelection(.enabled)
        }
    }
}
