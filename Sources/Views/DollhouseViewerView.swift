import Combine
import RealityKit
import SwiftUI
import UIKit
import simd

/// Standalone interactive 3D "Dollhouse" viewer rendering the baked LiDAR room model
/// with architectural surface shading, toggleable ceiling culling, 3D equipment pins,
/// orbit/pan/zoom gestures, and tap-to-fly-to camera inspection.
public struct DollhouseViewerView: View {
    public let scan: BakedRoomScan
    public let equipment: [TrackedEquipment3D]
    @ObservedObject var focusController: CameraFocusController
    public let showCeiling: Bool
    public let onSelectEquipment: (TrackedEquipment3D) -> Void

    @StateObject private var sceneCoordinator = DollhouseSceneCoordinator()

    public init(
        scan: BakedRoomScan,
        equipment: [TrackedEquipment3D],
        focusController: CameraFocusController,
        showCeiling: Bool = false,
        onSelectEquipment: @escaping (TrackedEquipment3D) -> Void
    ) {
        self.scan = scan
        self.equipment = equipment
        self.focusController = focusController
        self.showCeiling = showCeiling
        self.onSelectEquipment = onSelectEquipment
    }

    public var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.08, green: 0.10, blue: 0.15),
                    Color(red: 0.04, green: 0.05, blue: 0.08)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            if #available(iOS 18.0, *) {
                DollhouseRealityView(
                    sceneCoordinator: sceneCoordinator,
                    focusController: focusController,
                    scan: scan,
                    equipment: equipment,
                    showCeiling: showCeiling
                )
            } else {
                DollhouseLegacyARView(
                    sceneCoordinator: sceneCoordinator,
                    focusController: focusController,
                    scan: scan,
                    equipment: equipment,
                    showCeiling: showCeiling
                )
            }

            DollhouseGestureOverlay(
                focusController: focusController,
                equipment: equipment,
                onSelectEquipment: onSelectEquipment
            )
        }
    }
}

// MARK: - iOS 18+ RealityView Implementation

@available(iOS 18.0, *)
private struct DollhouseRealityView: View {
    @ObservedObject var sceneCoordinator: DollhouseSceneCoordinator
    @ObservedObject var focusController: CameraFocusController
    let scan: BakedRoomScan
    let equipment: [TrackedEquipment3D]
    let showCeiling: Bool

    var body: some View {
        RealityView { content in
            content.camera = .virtual
            sceneCoordinator.configureLightingIfNeeded()
            sceneCoordinator.syncScanGeometry(scan, showCeiling: showCeiling)
            sceneCoordinator.syncEquipmentMarkers(
                items: equipment,
                focusedID: focusController.focusedItemID,
                useBillboardComponent: true
            )
            sceneCoordinator.updateCameraTransform(focusController.virtualCameraTransform)

            if sceneCoordinator.rootEntity.parent == nil {
                content.add(sceneCoordinator.rootEntity)
            }
        } update: { _ in
            sceneCoordinator.syncScanGeometry(scan, showCeiling: showCeiling)
            sceneCoordinator.syncEquipmentMarkers(
                items: equipment,
                focusedID: focusController.focusedItemID,
                useBillboardComponent: true
            )
            sceneCoordinator.updateCameraTransform(focusController.virtualCameraTransform)
        }
    }
}

// MARK: - iOS 17 ARView(.nonAR) Fallback Implementation

private struct DollhouseLegacyARView: UIViewRepresentable {
    @ObservedObject var sceneCoordinator: DollhouseSceneCoordinator
    @ObservedObject var focusController: CameraFocusController
    let scan: BakedRoomScan
    let equipment: [TrackedEquipment3D]
    let showCeiling: Bool

    func makeCoordinator() -> FrameObserver {
        FrameObserver()
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.cameraMode = .nonAR
        arView.automaticallyConfigureSession = false
        arView.environment.background = .color(UIColor(red: 0.06, green: 0.08, blue: 0.12, alpha: 1.0))

        let rootAnchor = AnchorEntity(world: matrix_identity_float4x4)
        arView.scene.addAnchor(rootAnchor)
        sceneCoordinator.configureLightingIfNeeded()
        rootAnchor.addChild(sceneCoordinator.rootEntity)

        sceneCoordinator.syncScanGeometry(scan, showCeiling: showCeiling)
        sceneCoordinator.syncEquipmentMarkers(
            items: equipment,
            focusedID: focusController.focusedItemID,
            useBillboardComponent: false
        )
        sceneCoordinator.updateCameraTransform(focusController.virtualCameraTransform)

        context.coordinator.observeUpdates(
            on: arView,
            sceneCoordinator: sceneCoordinator,
            focusController: focusController
        )
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        sceneCoordinator.syncScanGeometry(scan, showCeiling: showCeiling)
        sceneCoordinator.syncEquipmentMarkers(
            items: equipment,
            focusedID: focusController.focusedItemID,
            useBillboardComponent: false
        )
        sceneCoordinator.updateCameraTransform(focusController.virtualCameraTransform)
    }

    @MainActor
    final class FrameObserver {
        private var subscription: (any Cancellable)?

        func observeUpdates(
            on arView: ARView,
            sceneCoordinator: DollhouseSceneCoordinator,
            focusController: CameraFocusController
        ) {
            subscription = arView.scene.subscribe(to: SceneEvents.Update.self) { [weak sceneCoordinator, weak focusController] _ in
                MainActor.assumeIsolated {
                    guard let sceneCoordinator, let focusController else { return }
                    sceneCoordinator.orientMarkerLabelsToward(
                        cameraPosition: focusController.virtualCameraTransform.translation
                    )
                }
            }
        }
    }
}

// MARK: - Dollhouse Scene Graph Coordinator

@MainActor
final class DollhouseSceneCoordinator: ObservableObject {
    let rootEntity = Entity()
    private let geometryRootEntity = Entity()
    private let markersRootEntity = Entity()
    private let lightingRootEntity = Entity()
    private let virtualCamera = PerspectiveCamera()

    private var loadedScanID: UUID?
    private var submeshEntitiesByCategory: [RoomSurfaceCategory: ModelEntity] = [:]
    private var markerEntitiesByID: [UUID: Entity] = [:]
    private var markerNodesByID: [UUID: EquipmentMarkerNode] = [:]
    private var hasConfiguredLighting = false

    init() {
        rootEntity.addChild(geometryRootEntity)
        rootEntity.addChild(markersRootEntity)
        rootEntity.addChild(lightingRootEntity)

        virtualCamera.camera.fieldOfViewInDegrees = 60
        virtualCamera.isEnabled = true
        rootEntity.addChild(virtualCamera)
    }

    func configureLightingIfNeeded() {
        guard !hasConfiguredLighting else { return }
        hasConfiguredLighting = true

        let keyLight = DirectionalLight()
        keyLight.light.intensity = 2000
        keyLight.light.color = .white
        keyLight.look(at: .zero, from: SIMD3<Float>(4.0, 9.0, 5.0), relativeTo: nil)

        let fillLight = DirectionalLight()
        fillLight.light.intensity = 900
        fillLight.light.color = UIColor(red: 0.78, green: 0.86, blue: 1.0, alpha: 1.0)
        fillLight.look(at: .zero, from: SIMD3<Float>(-5.0, 4.0, -4.0), relativeTo: nil)

        let upliftLight = DirectionalLight()
        upliftLight.light.intensity = 450
        upliftLight.light.color = UIColor(red: 0.90, green: 0.92, blue: 0.96, alpha: 1.0)
        upliftLight.look(at: .zero, from: SIMD3<Float>(0.0, -6.0, 0.0), relativeTo: nil)

        lightingRootEntity.addChild(keyLight)
        lightingRootEntity.addChild(fillLight)
        lightingRootEntity.addChild(upliftLight)
    }

    func updateCameraTransform(_ transform: Transform) {
        virtualCamera.isEnabled = true
        virtualCamera.transform = transform
    }

    func syncScanGeometry(_ scan: BakedRoomScan, showCeiling: Bool) {
        if loadedScanID != scan.id {
            loadedScanID = scan.id
            for entity in submeshEntitiesByCategory.values {
                entity.removeFromParent()
            }
            submeshEntitiesByCategory.removeAll()

            for submesh in scan.geometry.submeshes {
                guard let mesh = Self.buildMeshResource(from: submesh) else { continue }
                let material = Self.material(for: submesh.category)
                let entity = ModelEntity(mesh: mesh, materials: [material])
                geometryRootEntity.addChild(entity)
                submeshEntitiesByCategory[submesh.category] = entity
            }
        }

        if let ceilingEntity = submeshEntitiesByCategory[.ceiling] {
            ceilingEntity.isEnabled = showCeiling
        }
    }

    func syncEquipmentMarkers(
        items: [TrackedEquipment3D],
        focusedID: UUID?,
        useBillboardComponent: Bool
    ) {
        let activeIDs = Set(items.map(\.id))

        for (id, entity) in markerEntitiesByID where !activeIDs.contains(id) {
            entity.removeFromParent()
            markerEntitiesByID.removeValue(forKey: id)
            markerNodesByID.removeValue(forKey: id)
        }

        for item in items {
            let isFocused = (item.id == focusedID)
            if let existingEntity = markerEntitiesByID[item.id],
               let existingNode = markerNodesByID[item.id] {
                existingEntity.position = item.worldPosition
                existingNode.update(item: item, isFocused: isFocused)
            } else {
                let node = EquipmentMarkerNode(
                    item: item,
                    isFocused: isFocused,
                    useBillboardComponent: useBillboardComponent
                )
                markersRootEntity.addChild(node.root)
                markerEntitiesByID[item.id] = node.root
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
                relativeTo: nil
            )
        }
    }

    private static func buildMeshResource(from submesh: BakedSubmesh) -> MeshResource? {
        guard !submesh.positions.isEmpty,
              submesh.positions.count == submesh.normals.count,
              !submesh.indices.isEmpty else {
            return nil
        }

        var descriptor = MeshDescriptor(name: "Dollhouse_\(submesh.category.rawValue)")
        descriptor.positions = MeshBuffers.Positions(submesh.positions)
        descriptor.normals = MeshBuffers.Normals(submesh.normals)
        descriptor.primitives = .triangles(submesh.indices)
        return try? MeshResource.generate(from: [descriptor])
    }

    private static func material(for category: RoomSurfaceCategory) -> SimpleMaterial {
        switch category {
        case .floor:
            return SimpleMaterial(
                color: UIColor(red: 0.34, green: 0.38, blue: 0.44, alpha: 1.0),
                roughness: 0.85,
                isMetallic: false
            )
        case .wall:
            return SimpleMaterial(
                color: UIColor(red: 0.84, green: 0.86, blue: 0.90, alpha: 1.0),
                roughness: 0.92,
                isMetallic: false
            )
        case .furniture:
            return SimpleMaterial(
                color: UIColor(red: 0.68, green: 0.56, blue: 0.44, alpha: 1.0),
                roughness: 0.72,
                isMetallic: false
            )
        case .opening:
            return SimpleMaterial(
                color: UIColor(red: 0.44, green: 0.68, blue: 0.80, alpha: 1.0),
                roughness: 0.35,
                isMetallic: true
            )
        case .ceiling:
            return SimpleMaterial(
                color: UIColor(red: 0.90, green: 0.91, blue: 0.94, alpha: 1.0),
                roughness: 0.95,
                isMetallic: false
            )
        case .other:
            return SimpleMaterial(
                color: UIColor(red: 0.56, green: 0.60, blue: 0.65, alpha: 1.0),
                roughness: 0.88,
                isMetallic: false
            )
        }
    }
}

// MARK: - Multi-Touch Orbit / Pan / Zoom & Tap Pin Raycast Overlay

private struct DollhouseGestureOverlay: UIViewRepresentable {
    @ObservedObject var focusController: CameraFocusController
    let equipment: [TrackedEquipment3D]
    let onSelectEquipment: (TrackedEquipment3D) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(
            focusController: focusController,
            equipment: equipment,
            onSelectEquipment: onSelectEquipment
        )
    }

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        view.backgroundColor = .clear
        view.isMultipleTouchEnabled = true

        let orbitPan = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleOrbitPan(_:))
        )
        orbitPan.minimumNumberOfTouches = 1
        orbitPan.maximumNumberOfTouches = 1

        let panGesture = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTwoFingerPan(_:))
        )
        panGesture.minimumNumberOfTouches = 2
        panGesture.maximumNumberOfTouches = 2

        let pinchGesture = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )

        let tapGesture = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )

        view.addGestureRecognizer(orbitPan)
        view.addGestureRecognizer(panGesture)
        view.addGestureRecognizer(pinchGesture)
        view.addGestureRecognizer(tapGesture)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.focusController = focusController
        context.coordinator.equipment = equipment
        context.coordinator.onSelectEquipment = onSelectEquipment
    }

    @MainActor
    final class Coordinator: NSObject {
        var focusController: CameraFocusController
        var equipment: [TrackedEquipment3D]
        var onSelectEquipment: (TrackedEquipment3D) -> Void

        init(
            focusController: CameraFocusController,
            equipment: [TrackedEquipment3D],
            onSelectEquipment: @escaping (TrackedEquipment3D) -> Void
        ) {
            self.focusController = focusController
            self.equipment = equipment
            self.onSelectEquipment = onSelectEquipment
        }

        @objc func handleOrbitPan(_ recognizer: UIPanGestureRecognizer) {
            let translation = recognizer.translation(in: recognizer.view)
            recognizer.setTranslation(.zero, in: recognizer.view)

            let sensitivity: Float = 0.0085
            focusController.orbitBy(
                deltaYaw: -Float(translation.x) * sensitivity,
                deltaPitch: Float(translation.y) * sensitivity
            )
        }

        @objc func handleTwoFingerPan(_ recognizer: UIPanGestureRecognizer) {
            let translation = recognizer.translation(in: recognizer.view)
            recognizer.setTranslation(.zero, in: recognizer.view)

            let panScale = max(focusController.orbitDistance * 0.0018, 0.002)
            focusController.panBy(
                deltaRight: -Float(translation.x) * panScale,
                deltaUp: Float(translation.y) * panScale
            )
        }

        @objc func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            guard recognizer.state == .changed else { return }
            focusController.zoomBy(scaleFactor: Float(recognizer.scale))
            recognizer.scale = 1.0
        }

        @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
            guard let view = recognizer.view,
                  view.bounds.width > 1,
                  view.bounds.height > 1 else {
                return
            }

            let location = recognizer.location(in: view)
            guard let matched = hitTestEquipmentPin(
                tapPoint: location,
                viewSize: view.bounds.size
            ) else {
                return
            }

            onSelectEquipment(matched)
        }

        private func hitTestEquipmentPin(
            tapPoint: CGPoint,
            viewSize: CGSize
        ) -> TrackedEquipment3D? {
            let cameraMatrix = focusController.virtualCameraTransform.matrix
            let rayOrigin = CameraFocusController.extractPosition(from: cameraMatrix)

            let fovRadians: Float = 60.0 * .pi / 180.0
            let halfHeight = tan(fovRadians * 0.5)
            let aspect = Float(viewSize.width / viewSize.height)
            let halfWidth = halfHeight * aspect

            let ndcX = Float((tapPoint.x / viewSize.width) * 2.0 - 1.0)
            let ndcY = Float(1.0 - (tapPoint.y / viewSize.height) * 2.0)

            let localRayDir = simd_normalize(SIMD3<Float>(
                ndcX * halfWidth,
                ndcY * halfHeight,
                -1.0
            ))

            let rot = simd_float3x3(columns: (
                SIMD3<Float>(cameraMatrix.columns.0.x, cameraMatrix.columns.0.y, cameraMatrix.columns.0.z),
                SIMD3<Float>(cameraMatrix.columns.1.x, cameraMatrix.columns.1.y, cameraMatrix.columns.1.z),
                SIMD3<Float>(cameraMatrix.columns.2.x, cameraMatrix.columns.2.y, cameraMatrix.columns.2.z)
            ))
            let rayDir = simd_normalize(rot * localRayDir)

            var bestItem: TrackedEquipment3D?
            var bestDistanceAlongRay = Float.greatestFiniteMagnitude

            for item in equipment {
                // Account for both the marker sphere and elevated billboard label center.
                let centers = [
                    item.worldPosition,
                    item.worldPosition + SIMD3<Float>(0, 0.075, 0)
                ]

                for target in centers {
                    let toTarget = target - rayOrigin
                    let projection = simd_dot(toTarget, rayDir)
                    guard projection > 0.05 else { continue }

                    let closestPoint = rayOrigin + rayDir * projection
                    let perpendicularDistance = simd_length(target - closestPoint)
                    let pickRadius = max(0.14, projection * 0.085)

                    if perpendicularDistance <= pickRadius, projection < bestDistanceAlongRay {
                        bestDistanceAlongRay = projection
                        bestItem = item
                    }
                }
            }

            return bestItem
        }
    }
}
