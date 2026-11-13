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
        if #available(iOS 18.0, *) {
            ModernRealitySceneView(
                sessionManager: sessionManager,
                clusteringManager: clusteringManager,
                focusController: focusController,
                meshRenderStyle: meshRenderStyle
            )
        } else {
            LegacyARSceneView(
                sessionManager: sessionManager,
                clusteringManager: clusteringManager,
                focusController: focusController,
                meshRenderStyle: meshRenderStyle
            )
        }
    }
}

// MARK: - iOS 18+ RealityView Implementation

@available(iOS 18.0, *)
private struct ModernRealitySceneView: View {
    private struct SessionSyncTrigger: Equatable {
        let state: ARSessionState
        let generation: UInt64
    }

    @ObservedObject var sessionManager: ARSessionManager
    @ObservedObject var clusteringManager: EquipmentClusteringManager
    @ObservedObject var focusController: CameraFocusController
    var meshRenderStyle: MeshRenderStyle

    @StateObject private var coordinator = RealitySceneCoordinator()
    @State private var trackingSession = SpatialTrackingSession()
    @State private var hasBoundTrackingSession = false
    @State private var boundSessionGeneration: UInt64?

    var body: some View {
        RealityView { content in
            let shouldUseVirtual = focusController.isVirtualCameraActive
            content.camera = shouldUseVirtual ? .virtual : .spatialTracking
            coordinator.attachRootIfNeeded(to: content)
            coordinator.setVirtualCameraActive(
                shouldUseVirtual,
                transform: focusController.virtualCameraTransform
            )
            coordinator.syncMeshes(from: sessionManager, renderStyle: meshRenderStyle, useWireframeLines: true)
            coordinator.syncEquipmentMarkers(
                items: clusteringManager.trackedEquipment,
                focusedID: focusController.focusedItemID,
                useBillboardComponent: true
            )
        } update: { content in
            let shouldUseVirtual = focusController.isVirtualCameraActive
            content.camera = shouldUseVirtual ? .virtual : .spatialTracking
            coordinator.setVirtualCameraActive(
                shouldUseVirtual,
                transform: focusController.virtualCameraTransform
            )

            coordinator.syncMeshes(from: sessionManager, renderStyle: meshRenderStyle, useWireframeLines: true)
            coordinator.syncEquipmentMarkers(
                items: clusteringManager.trackedEquipment,
                focusedID: focusController.focusedItemID,
                useBillboardComponent: true
            )
        }
        .background(focusController.isVirtualCameraActive ? Color.black : Color.clear)
        .onChange(of: sessionManager.sessionState) { _, newState in
            if newState != .running {
                hasBoundTrackingSession = false
                boundSessionGeneration = nil
            }
        }
        .task(
            id: SessionSyncTrigger(
                state: sessionManager.sessionState,
                generation: sessionManager.sessionRunGeneration
            )
        ) {
            guard sessionManager.sessionState == .running else {
                hasBoundTrackingSession = false
                boundSessionGeneration = nil
                return
            }
            guard !hasBoundTrackingSession || boundSessionGeneration != sessionManager.sessionRunGeneration else {
                return
            }
            hasBoundTrackingSession = true
            boundSessionGeneration = sessionManager.sessionRunGeneration
            let spatialConfig = SpatialTrackingSession.Configuration(
                tracking: [.world, .plane],
                sceneUnderstanding: [],
                camera: .back
            )
            let arConfig = sessionManager.makeWorldTrackingConfiguration()
            _ = await trackingSession.run(
                spatialConfig,
                session: sessionManager.session,
                arConfiguration: arConfig
            )
        }
        .onChange(of: sessionManager.meshRevision) { _, _ in
            coordinator.syncMeshes(from: sessionManager, renderStyle: meshRenderStyle, useWireframeLines: true)
        }
        .onChange(of: meshRenderStyle) { _, newStyle in
            coordinator.updateMeshMaterials(renderStyle: newStyle, useWireframeLines: true)
        }
    }
}

// MARK: - iOS 17 Fallback ARView Implementation

private struct LegacyARSceneView: UIViewRepresentable {
    @ObservedObject var sessionManager: ARSessionManager
    @ObservedObject var clusteringManager: EquipmentClusteringManager
    @ObservedObject var focusController: CameraFocusController
    var meshRenderStyle: MeshRenderStyle

    func makeCoordinator() -> LegacyCoordinator {
        LegacyCoordinator()
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.automaticallyConfigureSession = false
        arView.session = sessionManager.session

        let rootAnchor = AnchorEntity(world: matrix_identity_float4x4)
        arView.scene.addAnchor(rootAnchor)
        rootAnchor.addChild(context.coordinator.sceneCoordinator.rootEntity)

        context.coordinator.sceneCoordinator.syncMeshes(
            from: sessionManager,
            renderStyle: meshRenderStyle,
            useWireframeLines: false
        )
        context.coordinator.sceneCoordinator.syncEquipmentMarkers(
            items: clusteringManager.trackedEquipment,
            focusedID: focusController.focusedItemID,
            useBillboardComponent: false
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
        }
        sceneCoord.setVirtualCameraActive(
            shouldUseVirtual,
            transform: focusController.virtualCameraTransform
        )
        sceneCoord.syncMeshes(
            from: sessionManager,
            renderStyle: meshRenderStyle,
            useWireframeLines: false
        )
        sceneCoord.syncEquipmentMarkers(
            items: clusteringManager.trackedEquipment,
            focusedID: focusController.focusedItemID,
            useBillboardComponent: false
        )
        sceneCoord.updateMeshMaterials(renderStyle: meshRenderStyle, useWireframeLines: false)
    }

    @MainActor
    final class LegacyCoordinator {
        let sceneCoordinator = RealitySceneCoordinator()
        private var frameSubscription: AnyCancellable?

        func observeFrameUpdates(
            on arView: ARView,
            sessionManager: ARSessionManager,
            focusController: CameraFocusController
        ) {
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
    let meshRootEntity = Entity()
    let markersRootEntity = Entity()
    let virtualCameraEntity = PerspectiveCamera()

    private var isAttached = false
    private var meshEntitiesByID: [UUID: ModelEntity] = [:]
    private var markerNodesByID: [UUID: EquipmentMarkerNode] = [:]
    private var activeMeshStyle: MeshRenderStyle?
    private var activeUseWireframeLines: Bool?
    private var cachedMeshMaterial: (any RealityKit.Material)?

    init() {
        rootEntity.addChild(meshRootEntity)
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

    @available(iOS 18.0, *)
    func attachRootIfNeeded(to content: RealityViewCameraContent) {
        guard !isAttached else { return }
        content.add(rootEntity)
        isAttached = true
    }

    func syncMeshes(
        from sessionManager: ARSessionManager,
        renderStyle: MeshRenderStyle,
        useWireframeLines: Bool
    ) {
        if renderStyle != activeMeshStyle || useWireframeLines != activeUseWireframeLines {
            updateMeshMaterials(renderStyle: renderStyle, useWireframeLines: useWireframeLines)
        }

        let changes = sessionManager.consumePendingMeshChanges()

        for removedID in changes.removed {
            if let entity = meshEntitiesByID.removeValue(forKey: removedID) {
                entity.removeFromParent()
            }
        }

        // Seed existing meshes on first attachment if the coordinator's mesh map is empty.
        let anchorsToBuild: [ARMeshAnchor]
        if meshEntitiesByID.isEmpty {
            let allAnchors = sessionManager.allMeshAnchors()
            anchorsToBuild = allAnchors.isEmpty ? changes.upserted : allAnchors
        } else {
            anchorsToBuild = changes.upserted
        }

        guard !anchorsToBuild.isEmpty else {
            return
        }

        let material = cachedMeshMaterial ?? makeMeshMaterial(
            renderStyle: renderStyle,
            useWireframeLines: useWireframeLines
        )
        for anchor in anchorsToBuild {
            guard let meshResource = Self.buildMeshResource(from: anchor.geometry) else {
                continue
            }
            if let existingEntity = meshEntitiesByID[anchor.identifier] {
                existingEntity.model?.mesh = meshResource
                existingEntity.model?.materials = [material]
                existingEntity.transform = Transform(matrix: anchor.transform)
            } else {
                let modelEntity = ModelEntity(mesh: meshResource, materials: [material])
                modelEntity.transform = Transform(matrix: anchor.transform)
                meshRootEntity.addChild(modelEntity)
                meshEntitiesByID[anchor.identifier] = modelEntity
            }
        }
    }

    func updateMeshMaterials(renderStyle: MeshRenderStyle, useWireframeLines: Bool) {
        guard renderStyle != activeMeshStyle || useWireframeLines != activeUseWireframeLines else {
            return
        }
        activeMeshStyle = renderStyle
        activeUseWireframeLines = useWireframeLines
        let material = makeMeshMaterial(renderStyle: renderStyle, useWireframeLines: useWireframeLines)
        cachedMeshMaterial = material
        for entity in meshEntitiesByID.values {
            entity.model?.materials = [material]
        }
    }

    func syncEquipmentMarkers(
        items: [TrackedEquipment3D],
        focusedID: UUID?,
        useBillboardComponent: Bool
    ) {
        let activeIDs = Set(items.map(\.id))

        for (id, node) in markerNodesByID where !activeIDs.contains(id) {
            node.root.removeFromParent()
            markerNodesByID.removeValue(forKey: id)
        }

        for item in items {
            let isFocused = (item.id == focusedID)
            if let existing = markerNodesByID[item.id] {
                existing.root.position = item.worldPosition
                existing.updateAppearance(
                    label: item.label,
                    observationCount: item.observationCount,
                    isFocused: isFocused
                )
            } else {
                let node = EquipmentMarkerNode(
                    item: item,
                    isFocused: isFocused,
                    useBillboardComponent: useBillboardComponent
                )
                markersRootEntity.addChild(node.root)
                markerNodesByID[item.id] = node
            }
        }
    }

    func orientMarkerLabelsToward(cameraPosition: simd_float3) {
        guard cameraPosition.x.isFinite,
              cameraPosition.y.isFinite,
              cameraPosition.z.isFinite else {
            return
        }
        for node in markerNodesByID.values {
            let labelWorldPos = node.labelContainer.position(relativeTo: nil)
            guard simd_distance(cameraPosition, labelWorldPos) > 1e-4 else {
                continue
            }
            node.labelContainer.look(
                at: cameraPosition,
                from: labelWorldPos,
                relativeTo: nil,
                forward: .positiveZ
            )
        }
    }

    private func makeMeshMaterial(
        renderStyle: MeshRenderStyle,
        useWireframeLines: Bool
    ) -> any RealityKit.Material {
        switch renderStyle {
        case .wireframe:
            var material = UnlitMaterial(color: UIColor.systemCyan.withAlphaComponent(0.5))
            material.blending = .transparent(opacity: .init(floatLiteral: 1.0))
            if useWireframeLines, #available(iOS 18.0, *) {
                material.triangleFillMode = .lines
            }
            return material
        case .occlusion:
            return OcclusionMaterial()
        }
    }

    /// Safely converts ARKit `ARMeshGeometry` vertex and triangle buffers into a RealityKit `MeshResource`,
    /// respecting buffer offsets, vertex strides, and 16-bit/32-bit index sizes.
    static func buildMeshResource(from geometry: ARMeshGeometry) -> MeshResource? {
        let vertexSource = geometry.vertices
        let faceElement = geometry.faces

        guard vertexSource.count > 0,
              vertexSource.stride >= MemoryLayout<Float>.stride * 3,
              faceElement.count > 0,
              faceElement.primitiveType == .triangle,
              faceElement.indexCountPerPrimitive == 3 else {
            return nil
        }

        let totalIndices = faceElement.count * faceElement.indexCountPerPrimitive
        let requiredVertexBytes = vertexSource.offset + (vertexSource.count - 1) * vertexSource.stride + MemoryLayout<Float>.stride * 3
        let requiredIndexBytes = totalIndices * faceElement.bytesPerIndex
        guard vertexSource.offset >= 0,
              totalIndices > 0,
              vertexSource.buffer.length >= requiredVertexBytes,
              faceElement.buffer.length >= requiredIndexBytes else {
            return nil
        }

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(vertexSource.count)

        let vertexBasePtr = vertexSource.buffer.contents()
        for i in 0..<vertexSource.count {
            let ptr = vertexBasePtr.advanced(by: vertexSource.offset + i * vertexSource.stride)
            let x = ptr.loadUnaligned(fromByteOffset: 0, as: Float.self)
            let y = ptr.loadUnaligned(fromByteOffset: MemoryLayout<Float>.stride, as: Float.self)
            let z = ptr.loadUnaligned(fromByteOffset: MemoryLayout<Float>.stride * 2, as: Float.self)
            positions.append(SIMD3<Float>(x, y, z))
        }

        var indices: [UInt32] = []
        indices.reserveCapacity(totalIndices)

        let maxValidIndex = UInt32(vertexSource.count)
        let indexBasePtr = faceElement.buffer.contents()
        if faceElement.bytesPerIndex == MemoryLayout<UInt32>.size {
            for i in 0..<totalIndices {
                let value = indexBasePtr.loadUnaligned(
                    fromByteOffset: i * MemoryLayout<UInt32>.stride,
                    as: UInt32.self
                )
                guard value < maxValidIndex else { return nil }
                indices.append(value)
            }
        } else if faceElement.bytesPerIndex == MemoryLayout<UInt16>.size {
            for i in 0..<totalIndices {
                let value = indexBasePtr.loadUnaligned(
                    fromByteOffset: i * MemoryLayout<UInt16>.stride,
                    as: UInt16.self
                )
                let expanded = UInt32(value)
                guard expanded < maxValidIndex else { return nil }
                indices.append(expanded)
            }
        } else {
            return nil
        }

        var descriptor = MeshDescriptor(name: "LiDARMeshChunk")
        descriptor.positions = MeshBuffers.Positions(positions)
        descriptor.primitives = .triangles(indices)
        return try? MeshResource.generate(from: [descriptor])
    }
}

// MARK: - 3D Billboard Equipment Marker Node

@MainActor
private final class EquipmentMarkerNode {
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

    func updateAppearance(label: String, observationCount: Int, isFocused: Bool) {
        let newTitle = Self.formatTitle(label: label, count: observationCount)
        if newTitle != currentTitle {
            currentTitle = newTitle
            let newMesh = Self.makeTextMesh(title: newTitle)
            textEntity.model?.mesh = newMesh
            backdropEntity.model?.mesh = Self.makeBackdropMesh(for: newMesh)
            Self.centerText(textEntity, backdrop: backdropEntity)
        }

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

    private static func formatTitle(label: String, count: Int) -> String {
        "\(label.capitalized) (\(count))"
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
