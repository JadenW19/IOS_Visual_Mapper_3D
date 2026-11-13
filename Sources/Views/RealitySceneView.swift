import ARKit
import Combine
import RealityKit
import SwiftUI
import UIKit
import simd

/// Display style for the real-time LiDAR reconstruction mesh.
public enum MeshRenderStyle: String, CaseIterable, Sendable {
    case wireframe = "Wireframe"
    case occlusion = "Occlusion"
}

/// Primary interactive 3D spatial view rendering the live LiDAR reconstruction mesh,
/// floating 3D equipment billboard markers, and smooth virtual camera focus transitions.
public struct RealitySceneView: View {
    @ObservedObject var sessionManager: ARSessionManager
    @ObservedObject var clusteringManager: EquipmentClusteringManager
    @ObservedObject var focusController: CameraFocusController
    var meshRenderStyle: MeshRenderStyle

    public init(
        sessionManager: ARSessionManager,
        clusteringManager: EquipmentClusteringManager,
        focusController: CameraFocusController,
        meshRenderStyle: MeshRenderStyle = .wireframe
    ) {
        self.sessionManager = sessionManager
        self.clusteringManager = clusteringManager
        self.focusController = focusController
        self.meshRenderStyle = meshRenderStyle
    }

    public var body: some View {
        ARSceneViewRepresentable(
            sessionManager: sessionManager,
            clusteringManager: clusteringManager,
            focusController: focusController,
            meshRenderStyle: meshRenderStyle
        )
    }
}

// MARK: - Unified ARView Implementation

private struct ARSceneViewRepresentable: UIViewRepresentable {
    @ObservedObject var sessionManager: ARSessionManager
    @ObservedObject var clusteringManager: EquipmentClusteringManager
    @ObservedObject var focusController: CameraFocusController
    var meshRenderStyle: MeshRenderStyle

    func makeCoordinator() -> ARViewCoordinator {
        ARViewCoordinator()
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.cameraMode = .ar
        arView.automaticallyConfigureSession = false
        arView.session = sessionManager.session
        arView.environment.background = .cameraFeed()

        let rootAnchor = AnchorEntity(world: matrix_identity_float4x4)
        arView.scene.addAnchor(rootAnchor)
        rootAnchor.addChild(context.coordinator.sceneCoordinator.rootEntity)

        let supportsBillboard: Bool
        if #available(iOS 18.0, *) {
            supportsBillboard = true
        } else {
            supportsBillboard = false
        }

        context.coordinator.sceneCoordinator.applyHardwareMeshStyle(meshRenderStyle, on: arView)
        context.coordinator.sceneCoordinator.syncLiveDots(
            candidates: clusteringManager.observationCandidates,
            finalizedEquipment: clusteringManager.trackedEquipment,
            focusedID: focusController.focusedItemID,
            useBillboardComponent: supportsBillboard
        )

        context.coordinator.observeFrameUpdates(
            on: arView,
            sessionManager: sessionManager,
            focusController: focusController
        )
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        let sceneCoord = context.coordinator.sceneCoordinator
        let shouldUseVirtual = focusController.isVirtualCameraActive
        let targetMode: ARView.CameraMode = shouldUseVirtual ? .nonAR : .ar
        if uiView.cameraMode != targetMode {
            uiView.cameraMode = targetMode
            uiView.environment.background = shouldUseVirtual ? .color(.black) : .cameraFeed()
        }

        let supportsBillboard: Bool
        if #available(iOS 18.0, *) {
            supportsBillboard = true
        } else {
            supportsBillboard = false
        }

        sceneCoord.setVirtualCameraActive(
            shouldUseVirtual,
            transform: focusController.virtualCameraTransform
        )
        sceneCoord.applyHardwareMeshStyle(meshRenderStyle, on: uiView)
        sceneCoord.syncLiveDots(
            candidates: clusteringManager.observationCandidates,
            finalizedEquipment: clusteringManager.trackedEquipment,
            focusedID: focusController.focusedItemID,
            useBillboardComponent: supportsBillboard
        )
    }

    @MainActor
    final class ARViewCoordinator {
        let sceneCoordinator = RealitySceneCoordinator()
        private var frameSubscription: (any Cancellable)?

        func observeFrameUpdates(
            on arView: ARView,
            sessionManager: ARSessionManager,
            focusController: CameraFocusController
        ) {
            if #available(iOS 18.0, *) {
                // iOS 18+ markers use BillboardComponent directly.
                return
            }
            frameSubscription = arView.scene.subscribe(to: SceneEvents.Update.self) { [weak self, weak sessionManager, weak focusController] _ in
                MainActor.assumeIsolated {
                    guard let self, let sessionManager, let focusController else { return }
                    let activeCameraPos = focusController.isVirtualCameraActive
                        ? focusController.virtualCameraTransform.translation
                        : sessionManager.currentCameraPosition
                    self.sceneCoordinator.orientMarkerLabelsToward(cameraPosition: activeCameraPos)
                }
            }
        }
    }
}

// MARK: - Shared RealityKit Scene Graph Coordinator

@MainActor
final class RealitySceneCoordinator: ObservableObject {
    let rootEntity = Entity()
    let markersRootEntity = Entity()
    let virtualCameraEntity = PerspectiveCamera()

    private var candidateDotEntitiesByID: [UUID: ModelEntity] = [:]
    private var focusedMarkerNode: EquipmentMarkerNode?
    private var focusedMarkerItemID: UUID?
    private var activeMeshStyle: MeshRenderStyle?

    // Single shared sphere mesh + material across all live scan dots so GPU memory stays O(1).
    private let sharedDotMesh: MeshResource = MeshResource.generateSphere(radius: 0.015)
    private let sharedDotMaterial = UnlitMaterial(color: UIColor.systemCyan.withAlphaComponent(0.78))
    private let maxLiveIndicatorDots: Int = 180

    init() {
        rootEntity.addChild(markersRootEntity)

        virtualCameraEntity.camera.fieldOfViewInDegrees = 60
        virtualCameraEntity.isEnabled = false
    }

    func setVirtualCameraActive(_ isActive: Bool, transform: Transform) {
        if isActive {
            if virtualCameraEntity.parent !== rootEntity {
                rootEntity.addChild(virtualCameraEntity)
            }
            virtualCameraEntity.isEnabled = true
            virtualCameraEntity.transform = transform
        } else {
            virtualCameraEntity.isEnabled = false
            if virtualCameraEntity.parent != nil {
                virtualCameraEntity.removeFromParent()
            }
        }
    }

    /// Uses ARView's built-in Metal hardware scene-understanding overlay rather than
    /// duplicating `ARMeshAnchor` geometry buffers in RealityKit on every frame.
    func applyHardwareMeshStyle(_ renderStyle: MeshRenderStyle, on arView: ARView) {
        guard renderStyle != activeMeshStyle else { return }
        activeMeshStyle = renderStyle

        switch renderStyle {
        case .wireframe:
            arView.environment.sceneUnderstanding.options.remove(.occlusion)
            arView.debugOptions.insert(.showSceneUnderstanding)
        case .occlusion:
            arView.debugOptions.remove(.showSceneUnderstanding)
            arView.environment.sceneUnderstanding.options.insert(.occlusion)
        }
    }

    /// Renders lightweight single-sphere indicator dots for captured candidates without
    /// spawning expensive 3D extruded `MeshResource.generateText` meshes during live capture.
    func syncLiveDots(
        candidates: [ObservationCandidate3D],
        finalizedEquipment: [TrackedEquipment3D],
        focusedID: UUID?,
        useBillboardComponent: Bool
    ) {
        // Show downsampled recent candidate dots if scanning in candidate mode, or equipment dots.
        let recentCandidates = Array(candidates.suffix(maxLiveIndicatorDots))
        let dotEntries: [(id: UUID, position: simd_float3)] = recentCandidates.isEmpty
            ? finalizedEquipment.prefix(maxLiveIndicatorDots).map { ($0.id, $0.worldPosition) }
            : recentCandidates.map { ($0.id, $0.position) }

        let activeIDs = Set(dotEntries.map(\.id))
        for (id, entity) in candidateDotEntitiesByID where !activeIDs.contains(id) {
            entity.removeFromParent()
            candidateDotEntitiesByID.removeValue(forKey: id)
        }

        for entry in dotEntries {
            if let existing = candidateDotEntitiesByID[entry.id] {
                existing.position = entry.position
            } else {
                let dot = ModelEntity(mesh: sharedDotMesh, materials: [sharedDotMaterial])
                dot.position = entry.position
                markersRootEntity.addChild(dot)
                candidateDotEntitiesByID[entry.id] = dot
            }
        }

        if let focusedID,
           let focusedItem = finalizedEquipment.first(where: { $0.id == focusedID }) {
            if focusedMarkerItemID == focusedID, let node = focusedMarkerNode {
                node.root.position = focusedItem.worldPosition
                node.update(item: focusedItem, isFocused: true)
            } else {
                focusedMarkerNode?.root.removeFromParent()
                let node = EquipmentMarkerNode(
                    item: focusedItem,
                    isFocused: true,
                    useBillboardComponent: useBillboardComponent
                )
                markersRootEntity.addChild(node.root)
                focusedMarkerNode = node
                focusedMarkerItemID = focusedID
            }
        } else {
            focusedMarkerNode?.root.removeFromParent()
            focusedMarkerNode = nil
            focusedMarkerItemID = nil
        }
    }

    func orientMarkerLabelsToward(cameraPosition: simd_float3) {
        guard let node = focusedMarkerNode,
              cameraPosition.x.isFinite,
              cameraPosition.y.isFinite,
              cameraPosition.z.isFinite else {
            return
        }
        let labelWorldPos = node.labelContainer.position(relativeTo: nil)
        guard simd_distance(cameraPosition, labelWorldPos) > 1e-4 else {
            return
        }
        if #available(iOS 18.0, *) {
            node.labelContainer.look(
                at: cameraPosition,
                from: labelWorldPos,
                relativeTo: nil,
                forward: .positiveZ
            )
        } else {
            node.labelContainer.look(
                at: cameraPosition,
                from: labelWorldPos,
                relativeTo: nil
            )
        }
    }
}

// MARK: - 3D Billboard Equipment Marker Node

@MainActor
final class EquipmentMarkerNode {
    let root = Entity()
    let sphereEntity: ModelEntity
    let labelContainer = Entity()
    private var textEntity: ModelEntity
    private var backdropEntity: ModelEntity

    private var currentTitle: String
    private var currentFocused: Bool

    init(item: TrackedEquipment3D, isFocused: Bool, useBillboardComponent: Bool) {
        self.currentTitle = Self.formatTitle(label: item.label, count: item.observationCount)
        self.currentFocused = isFocused

        let sphereMesh = MeshResource.generateSphere(radius: 0.024)
        let sphereMaterial = UnlitMaterial(color: isFocused ? .systemYellow : .systemTeal)
        self.sphereEntity = ModelEntity(mesh: sphereMesh, materials: [sphereMaterial])

        let textMesh = Self.makeTextMesh(title: currentTitle)
        let textMaterial = UnlitMaterial(color: .white)
        self.textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])

        let backdropMesh = Self.makeBackdropMesh(for: textMesh)
        var backdropMaterial = UnlitMaterial(
            color: isFocused
                ? UIColor(red: 0.28, green: 0.22, blue: 0.02, alpha: 0.88)
                : UIColor.black.withAlphaComponent(0.75)
        )
        backdropMaterial.blending = .transparent(opacity: .init(floatLiteral: 1.0))
        self.backdropEntity = ModelEntity(mesh: backdropMesh, materials: [backdropMaterial])

        root.position = item.worldPosition
        root.addChild(sphereEntity)

        labelContainer.position = SIMD3<Float>(0, 0.075, 0)
        if useBillboardComponent, #available(iOS 18.0, *) {
            labelContainer.components.set(BillboardComponent())
        }

        Self.centerText(textEntity, backdrop: backdropEntity)
        labelContainer.addChild(backdropEntity)
        labelContainer.addChild(textEntity)
        root.addChild(labelContainer)

        applyFocusScale(isFocused: isFocused)
    }

    func update(item: TrackedEquipment3D, isFocused: Bool) {
        let updatedTitle = Self.formatTitle(label: item.label, count: item.observationCount)
        if updatedTitle != currentTitle {
            currentTitle = updatedTitle
            let textMesh = Self.makeTextMesh(title: updatedTitle)
            textEntity.model?.mesh = textMesh
            backdropEntity.model?.mesh = Self.makeBackdropMesh(for: textMesh)
            Self.centerText(textEntity, backdrop: backdropEntity)
        }
        updateFocusState(isFocused: isFocused)
    }

    func updateFocusState(isFocused: Bool) {
        if isFocused != currentFocused {
            currentFocused = isFocused
            sphereEntity.model?.materials = [UnlitMaterial(color: isFocused ? .systemYellow : .systemTeal)]
            var backdropMaterial = UnlitMaterial(
                color: isFocused
                    ? UIColor(red: 0.28, green: 0.22, blue: 0.02, alpha: 0.88)
                    : UIColor.black.withAlphaComponent(0.75)
            )
            backdropMaterial.blending = .transparent(opacity: .init(floatLiteral: 1.0))
            backdropEntity.model?.materials = [backdropMaterial]
            applyFocusScale(isFocused: isFocused)
        }
    }

    private func applyFocusScale(isFocused: Bool) {
        let scale: Float = isFocused ? 1.35 : 1.0
        sphereEntity.scale = SIMD3<Float>(repeating: scale)
    }

    private static func formatTitle(label: String, count _: Int) -> String {
        label.capitalized
    }

    private static func makeTextMesh(title: String) -> MeshResource {
        MeshResource.generateText(
            title,
            extrusionDepth: 0.002,
            font: .systemFont(ofSize: 0.028, weight: .semibold),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
    }

    private static func makeBackdropMesh(for textMesh: MeshResource) -> MeshResource {
        let bounds = textMesh.bounds.extents
        let width = max(bounds.x + 0.028, 0.08)
        let height = max(bounds.y + 0.018, 0.036)
        return MeshResource.generatePlane(width: width, height: height, cornerRadius: height * 0.35)
    }

    private static func centerText(_ textEntity: ModelEntity, backdrop: ModelEntity) {
        guard let mesh = textEntity.model?.mesh else { return }
        let bounds = mesh.bounds
        let center = bounds.center
        textEntity.position = SIMD3<Float>(-center.x, -center.y, 0.002)
        backdrop.position = SIMD3<Float>(0, 0, -0.002)
    }
}
